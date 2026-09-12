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
from plugins.wisdom.client import WisdomClient, WisdomError, new_installation_id
from plugins.wisdom.package import PackageError, prepare, slug_for

Confirm = Callable[[str, str], bool]

_ORG_DIR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class NotConfirmed(WisdomError):
    pass


def _org_dir(org_id: str) -> str:
    return org_id if _ORG_DIR_RE.fullmatch(org_id) else "org-" + hashlib.sha256(org_id.encode()).hexdigest()


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

    def status(self) -> dict:
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
        return {"org_id": self.client.org_id, "installed": ledger, "updates": updates}

    # --- install / update / uninstall -------------------------------------------------------
    def install(self, skill_id: str, *, version: int | None, confirm: Confirm) -> dict:
        detail = self.client.skill(skill_id)
        skill = detail["skill"]
        if skill.get("state") != "active":
            raise WisdomError(f"skill is {skill.get('state')}; only active skills install")
        target_version = version or max((v["version"] for v in detail.get("versions") or []), default=None)
        if not target_version:
            raise WisdomError("skill has no published version")
        meta = self.client.version(skill_id, target_version)
        v = meta["version"]
        security = (v.get("security_check") or {})
        if security.get("status") == "blocked":
            raise WisdomError("Gateway security check blocked this version")
        slug = skill.get("slug") or skill_id
        title = f"Install Wisdom skill {slug} v{target_version}"
        detail_text = _text({"skill": skill_id, "version": target_version, "content_hash": v.get("content_hash"),
                             "security": f"{security.get('status')} — {security.get('summary', '')}",
                             "author": v.get("author_description"), "explanation": v.get("explanation"),
                             "target": self._root() / slug})
        if not confirm(title, detail_text):
            raise NotConfirmed("install not confirmed")
        ident = self.installation_id()
        generation = int(skill.get("takedown_generation", 0))
        chash, files = self.client.content(skill_id, target_version, installation_id=ident, takedown_generation=generation)
        if chash != v.get("content_hash"):
            raise PackageError("downloaded content does not match the version the user reviewed")
        dest = self._root() / slug
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".wisdom-", dir=str(dest.parent)))
        for rel, _, body in files:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_bytes(body)
        record = self.client.record_install(skill_id=skill_id, installation_id=ident, version=target_version,
                                            takedown_generation=generation)
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
        ledger = self._ledger()
        ledger[skill_id] = {"slug": slug, "version": target_version, "content_hash": chash, "path": str(dest),
                            "update_mode": record.get("effective_update_mode")}
        self.state.set("installed", ledger)
        return {"installed": skill_id, "slug": slug, "version": target_version, "path": str(dest)}

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
            prepared = prepare(source, description=description, owner="owner",
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
        import time
        deadline = time.monotonic() + timeout
        while draft.get("state") == "vetting" and time.monotonic() < deadline:
            time.sleep(2)
            draft = self.client.draft(draft["id"])
        return draft
