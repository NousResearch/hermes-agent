"""Wisdom actions: browse, install, update, uninstall, share — one function per verb.

Installed packages live under ``<skills>/_wisdom/<org>/<slug>/`` so the normal skill index picks
them up; the ledger in plugin state remembers exact versions so updates and uninstalls are
hash-bound to what the user consented to. Every mutating verb takes ``confirm``: a callable
returning True once a human has seen ``(title, detail)``. Callers pick the surface (CLI prompt,
approval gate); the service never applies without it.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_skills_dir
from plugins.wisdom import notices
from plugins.wisdom.client import WisdomClient, WisdomError, new_installation_id
from plugins.wisdom.package import PackageError, content_hash, prepare, sha256_address, slug_for

Confirm = Callable[[str, str], bool]

_ORG_DIR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class NotConfirmed(WisdomError):
    pass


def _org_dir(org_id: str) -> str:
    return org_id if _ORG_DIR_RE.fullmatch(org_id) else "org-" + hashlib.sha256(org_id.encode()).hexdigest()


def _swap_in(staged: Path, dest: Path, *, park: Path, installed_hash: str | None) -> Path | None:
    """Replace ``dest`` with ``staged`` by moves. The old tree is parked under plugin state (outside the
    skills tree, so the scanner never sees two copies) until the new one is in place: an interruption
    leaves the old or the new skill installed, never neither. When the old tree no longer matches the
    hash the user installed, it carries local edits and is kept; its path is returned."""
    backup = None
    if dest.exists():
        park.mkdir(parents=True, exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=f"{dest.name}-", dir=str(park)))
        backup.rmdir()
        shutil.move(str(dest), str(backup))
    try:
        shutil.move(str(staged), str(dest))
    except BaseException:
        if backup is not None and not dest.exists():
            shutil.move(str(backup), str(dest))
        raise
    if backup is None:
        return None
    if installed_hash and _hash_tree(backup) == installed_hash:
        shutil.rmtree(backup, ignore_errors=True)
        return None
    return backup


def _hash_tree(root: Path) -> str:
    return content_hash([(p.relative_to(root).as_posix(), sha256_address(p.read_bytes()))
                         for p in root.rglob("*") if p.is_file()])


def _text(fields: dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in fields.items() if v not in (None, "", []))


class Wisdom:
    def __init__(self, state, client: WisdomClient | None = None):
        self.state = state
        self.client = client or WisdomClient()

    # --- identity + ledger -------------------------------------------------------------------
    def installation_id(self) -> str:
        """Random per-profile consumer identity; registered with the Gateway on first use."""
        ident = self.state.get("installation_id")
        if not ident:
            ident = new_installation_id()
            self.client.register_identity(ident)
            self.state.set("installation_id", ident)
        return ident

    def _ledger(self) -> dict[str, dict]:
        return dict(self.state.get("installed") or {})

    def _root(self) -> Path:
        org = self.client.org_id
        if not org:
            raise WisdomError("Your Nous token carries no organization; Collective Wisdom is team-scoped")
        return get_skills_dir() / "_wisdom" / _org_dir(org)

    # --- read verbs ------------------------------------------------------------------------
    def browse(self) -> list[dict]:
        out, cursor = [], None
        while True:
            page = self.client.list_skills(cursor=cursor)
            for s in page.get("skills") or []:
                if s.get("state") == "active":
                    out.append({"id": s["id"], "slug": s.get("slug"), "version": s.get("latest_version"),
                                "installs": s.get("install_count", 0), "description": s.get("author_description"),
                                "security": (s.get("security_check") or {}).get("status")})
            cursor = page.get("next_cursor")
            if not cursor or len(out) >= 500:
                return out

    def show(self, skill_id: str) -> dict:
        detail = self.client.skill(skill_id)
        versions = detail.get("versions") or []
        latest = versions[-1] if versions else {}
        installed = self._ledger().get(skill_id)
        return {"skill": detail.get("skill"), "latest": latest,
                "installed_version": installed and installed["version"], "versions": [v.get("version") for v in versions]}

    def status(self, *, include_paths: bool = True) -> dict:
        """``include_paths=False`` for shared surfaces (gateway chats): local filesystem layout stays local."""
        ledger = self._ledger()
        updates = []
        if ledger:
            for row in self.client.installations(self.installation_id()):
                local = ledger.get(row["skill_id"])
                latest = row.get("latest_version")
                if local and latest and latest > local["version"]:
                    updates.append({"skill_id": row["skill_id"], "slug": local["slug"],
                                    "installed": local["version"], "latest": latest,
                                    "required": row.get("update_mode") == "REQUIRED"})
        shown = ledger if include_paths else {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in ledger.items()}
        return {"org_id": self.client.org_id, "installed": shown, "updates": updates, **notices.summary(self.state)}

    # --- install / update / uninstall -------------------------------------------------------
    def plan(self, skill_id: str, *, version: int | None = None) -> dict:
        """Everything a human must see before an install: exact version, hashes, Gateway verdict."""
        detail = self.client.skill(skill_id)
        skill = detail["skill"]
        if skill.get("state") != "active":
            raise WisdomError(f"skill is {skill.get('state')}; only active skills install")
        target = version or max((v["version"] for v in detail.get("versions") or []), default=None)
        if not target:
            raise WisdomError("skill has no published version")
        v = self.client.version(skill_id, target)["version"]
        security = v.get("security_check") or {}
        if security.get("status") == "blocked":
            raise WisdomError("Gateway security check blocked this version")
        slug = skill.get("slug") or skill_id
        return {"skill_id": skill_id, "slug": slug, "version": target, "content_hash": v.get("content_hash"),
                "takedown_generation": int(skill.get("takedown_generation", 0)),
                "security": f"{security.get('status')} — {security.get('summary', '')}",
                "author": v.get("author_description"), "explanation": v.get("explanation"),
                "target": str(self._root() / slug)}

    def install(self, skill_id: str, *, version: int | None, confirm: Confirm) -> dict:
        p = self.plan(skill_id, version=version)
        title = f"Install Wisdom skill {p['slug']} v{p['version']}"
        shown = {k: p[k] for k in ("skill_id", "version", "content_hash", "security", "author", "explanation", "target")}
        if not confirm(title, _text(shown)):
            raise NotConfirmed("install not confirmed")
        ident = self.installation_id()
        chash, files = self.client.content(skill_id, p["version"], installation_id=ident,
                                           takedown_generation=p["takedown_generation"])
        if chash != p["content_hash"]:
            raise PackageError("downloaded content does not match the version the user reviewed")
        dest = Path(p["target"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Stage under plugin state, never inside skills/: the skill scanner rglobs SKILL.md, so a
        # half-written tree in the skills dir would be discoverable before the user's consent lands.
        tmp = Path(tempfile.mkdtemp(prefix="install-", dir=str(self.state.data_dir)))
        try:
            for rel, _, body in files:
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                (tmp / rel).write_bytes(body)
            # Network first, local swap last: the only step left after the Gateway accepts is a move.
            record = self.client.record_install(skill_id=skill_id, installation_id=ident, version=p["version"],
                                                takedown_generation=p["takedown_generation"])
            previous = self._ledger().get(skill_id) or {}
            kept = _swap_in(tmp, dest, park=self.state.data_dir / "local-edits",
                            installed_hash=previous.get("content_hash"))
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        ledger = self._ledger()
        ledger[skill_id] = {"slug": p["slug"], "version": p["version"], "content_hash": chash, "path": str(dest),
                            "update_mode": record.get("effective_update_mode")}
        self.state.set("installed", ledger)
        notices.dismiss(self.state, skill_id)
        out = {"installed": skill_id, "slug": p["slug"], "version": p["version"], "path": str(dest)}
        if kept is not None:
            out["preserved_local_edits"] = str(kept)
        return out

    def update(self, skill_id: str | None, *, confirm: Confirm) -> list[dict]:
        pending = self.status()["updates"]
        if skill_id:
            pending = [u for u in pending if u["skill_id"] == skill_id or u["slug"] == skill_id]
        return [self.install(u["skill_id"], version=u["latest"], confirm=confirm) for u in pending]

    def uninstall(self, skill_id: str, *, confirm: Confirm) -> dict:
        ledger = self._ledger()
        key = next((k for k, v in ledger.items() if k == skill_id or v["slug"] == skill_id), None)
        if key is None:
            raise WisdomError(f"{skill_id} is not a Wisdom-managed installation")
        entry = ledger[key]
        path = Path(entry["path"]).resolve()
        if not path.is_relative_to(self._root().resolve()):
            raise WisdomError("managed path escaped the Wisdom root; refusing to delete")
        if not confirm(f"Uninstall Wisdom skill {entry['slug']}", _text({"skill": key, "path": path})):
            raise NotConfirmed("uninstall not confirmed")
        self.client.deactivate(self.installation_id(), key)
        shutil.rmtree(path, ignore_errors=True)
        del ledger[key]
        self.state.set("installed", ledger)
        notices.dismiss(self.state, key)
        return {"uninstalled": key, "slug": entry["slug"]}

    # --- share -------------------------------------------------------------------------------
    def share(self, skill_name: str, *, description: str, confirm: Confirm) -> dict:
        """Package a local skill, review it, upload as an owner-private draft, then approve+publish.
        Two confirmations: the package (bytes + description) before upload, and the Gateway's own
        review (its security/professionalism verdicts) before publication."""
        from tools.skill_usage import _find_skill_dir
        source = _find_skill_dir(skill_name)
        if source is None:
            raise WisdomError(f"local skill {skill_name!r} not found")
        if source.resolve().is_relative_to((get_skills_dir() / "_wisdom").resolve()):
            raise WisdomError("a Wisdom-managed installation cannot be re-shared; fork it first")
        slug = slug_for(skill_name)
        with tempfile.TemporaryDirectory(prefix="wisdom-share-") as staging:
            prepared = prepare(source, description=description, owner=self.client.owner,
                               installation_id=self.installation_id(), staging=Path(staging))
            listing = "\n".join(f"  {p} ({len(b)} bytes)" for p, _, b in prepared.files)
            if not confirm(f"Share {skill_name} with your team as {slug}",
                           f"files:\n{listing}\ncontent_hash: {prepared.content_hash}\n"
                           f"description: {prepared.description}"):
                raise NotConfirmed("share not confirmed")
            draft = self.client.submit_draft(prepared, slug=slug)
        draft = self._await_vetting(draft)
        if draft.get("state") != "ready":
            return {"draft_id": draft["id"], "state": draft.get("state"), "note": draft.get("moderationNote"),
                    "security": draft.get("security_check")}
        for field, local in (("contentHash", prepared.content_hash), ("authorDescriptionHash", prepared.description_hash),
                             ("packageManifestHash", prepared.manifest_hash)):
            if draft.get(field) != local:
                raise PackageError(f"Gateway {field} differs from the reviewed package; not publishing")
        sec, prof = draft.get("security_check") or {}, draft.get("professionalism_check") or {}
        if not confirm(f"Publish {slug} to your team",
                       _text({"draft": draft["id"], "security": f"{sec.get('status')} — {sec.get('summary', '')}",
                              "professionalism": f"{prof.get('status')} — {prof.get('summary', '')}",
                              "content_hash": prepared.content_hash})):
            self.client.decline(draft["id"])
            raise NotConfirmed("publication declined; draft withdrawn")
        result = self.client.approve_and_publish(draft["id"], content_hash=prepared.content_hash,
                                                 description_hash=prepared.description_hash,
                                                 manifest_hash=prepared.manifest_hash)
        return {"draft_id": draft["id"], "skill_id": result.get("skill_id"), "version": result.get("version"),
                "outcome": result.get("publication_outcome"), "message": result.get("user_message"),
                "review_url": result.get("review_url")}

    def _await_vetting(self, draft: dict, *, timeout: float = 90.0) -> dict:
        deadline = time.monotonic() + timeout
        while draft.get("state") == "vetting" and time.monotonic() < deadline:
            time.sleep(2)
            draft = self.client.draft(draft["id"])
        return draft
