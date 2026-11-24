"""The /sesh + /nitesesh interactive Discord panel (rebuild of the Nov 2026 UI).

Lives as a standalone file on purpose: Hermes updates never touch it. The only
integration point is a try/except hook in DiscordAdapter._register_slash_commands
(see Z:/HERMES/.hermes/nitesesh/REAPPLY.md). If this file is missing the panel is
gone; if the hook is dropped the panel is gone. Restore both after `hermes update`.

Panel behaviour (user spec 2026-11-24):
  /sesh      -> ONE embed listing EVERY project (numbered), no pagination.
                3 project modals (1-15 / 16-30 / 31+) tick/untick by number.
                Settings, Model and Fallbacks each open their own modal.
  /nitesesh  -> status embed.
Views never exceed 5 rows (Discord hard cap) - the old 7-row SettingsView threw
on send, which is why the Settings button appeared dead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import discord
except ImportError:  # pragma: no cover - adapter only loads this with discord present
    discord = None

logger = logging.getLogger(__name__)

SETTINGS_PATH = "Z:/HERMES/.hermes/nitesesh/settings.json"
DISCOVER_SCRIPT = "Z:/HERMES/.hermes/scripts/discover-todo-projects.py"
RUNNER_SCRIPT = "Z:/HERMES/.hermes/scripts/nitesesh-run.py"
RUNS_DIR = "Z:/HERMES/.hermes/nitesesh/runs"
MODAL_SPAN = 15  # projects covered by each project modal (3 modals = 45)


# ---------------------------------------------------------------------------
# project enumeration: list EVERY git repo under roots + extra_projects,
# whether or not it has a todo file (the discovery script drops no-todo repos).
# ---------------------------------------------------------------------------

def _load_settings() -> dict[str, Any]:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover
        logger.warning("nitesesh_ui: cannot read settings: %s", exc)
        return {}


def _save_settings(settings: dict[str, Any]) -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


def _git_out(path: str, *args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", path, *args], check=False, capture_output=True,
                           text=True, timeout=15)
        return r.stdout.strip()
    except Exception:
        return ""


def _age_label(days_ago: Optional[int]) -> str:
    if days_ago is None:
        return "no commits"
    if days_ago <= 0:
        return "just now"
    if days_ago == 1:
        return "1d ago"
    return f"{days_ago}d ago"


def _scan_projects() -> list[dict[str, Any]]:
    """Full project list: every git repo under roots (recursive) + extra_projects
    + saved names, with open-task counts, last-commit age and dirty markers."""
    cfg = _load_settings()
    roots = cfg.get("roots") or ["Z:/github"]
    extra = cfg.get("extra_projects") or []
    saved_names = [n for n in (cfg.get("projects") or []) if not str(n).startswith("##")]

    skip_names = {"node_modules", "__pycache__", "dist", "build", "release",
                  ".git", ".venv", "venv", "build-target", "target", "assets"}

    # 1. recursive scan for git repos (depth<=4)
    found: dict[str, Path] = {}
    for r in roots:
        root = Path(r)
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).parts) - base_depth
            dirnames[:] = [d for d in dirnames
                           if d not in skip_names and not d.startswith(".")]
            if depth > 4:
                dirnames[:] = []
                continue
            here = Path(dirpath)
            if (here / ".git").exists():
                found[str(here).lower()] = here
                dirnames[:] = [d for d in dirnames if d not in ("plugins", "crates", "src", "lib")]

    # 2. extra_projects + saved settings names resolved to folders
    for path in list(extra) + [None] * len(saved_names):
        if path is not None:
            pp = Path(path)
            if pp.is_dir():
                found[str(pp).lower()] = pp
    for name in saved_names:
        leaf = name.split("·")[-1].strip() if "·" in name else name
        for r in roots:
            root = Path(r)
            if not root.is_dir():
                continue
            for dirpath, dirnames, _ in os.walk(root):
                dirnames[:] = [d for d in dirnames
                               if d not in skip_names and not d.startswith(".")]
                if Path(dirpath).name.lower() == leaf.lower():
                    found[str(dirpath).lower()] = Path(dirpath)
                if len(Path(dirpath).parts) - len(root.parts) > 3:
                    dirnames[:] = []

    # 3. open-task counts from the discovery script (single call, path-keyed)
    open_map: dict[str, int] = {}
    try:
        out = subprocess.run([sys.executable, DISCOVER_SCRIPT],
                             capture_output=True, text=True, timeout=180, check=True)
        for item in json.loads(out.stdout).get("projects", []):
            open_map[str(Path(item["path"])).lower()] = int(item.get("open_tasks", 0))
    except Exception as exc:
        logger.warning("nitesesh_ui: discovery lookup failed: %s", exc)

    # 4. git status + last-commit age per project
    out: list[dict[str, Any]] = []
    for key, proj in sorted(found.items()):
        is_git = (proj / ".git").exists()
        open_n = open_map.get(str(proj).lower(), 0)
        dirty, days = False, None
        if is_git:
            dirty = bool(_git_out(str(proj), "status", "--porcelain", "--untracked-files=no"))
            ts = _git_out(str(proj), "log", "-1", "--format=%ct")
            if ts.isdigit():
                days = int(time.time() - int(ts)) // 86400
        out.append({"name": proj.name, "path": str(proj), "open_tasks": open_n,
                    "is_git": is_git, "dirty": dirty, "last_commit_days": days})

    # 5. sort: recent activity first, no-commit / non-git at the bottom
    def sort_key(item):
        if not item["is_git"]:
            return (2, 0)
        if item["last_commit_days"] is None:
            return (1, 0)
        return (0, item["last_commit_days"])
    out.sort(key=sort_key)

    # 6. dedupe: same repo can appear under several names/paths (case, short
    #    clones, junctions). Keep the most recently committed entry per
    #    case-folded name, then re-sort so numbering stays stable.
    best: dict[str, dict[str, Any]] = {}
    for item in out:
        key = item["name"].casefold()
        prev = best.get(key)
        if prev is None or sort_key(item) < sort_key(prev):
            best[key] = item
    out = list(best.values())
    out.sort(key=sort_key)
    return out


def _mask(name: str, reveal: bool) -> str:
    """mask_names=true hides project names unless Reveal names is on."""
    if reveal or not name:
        return name
    clean = name.replace("·", " ").strip()
    parts = [p for p in clean.split() if p]
    if len(parts) > 1:
        return "_".join(p[:4].upper() for p in parts)
    raw = parts[0] if parts else name
    if len(raw) < 8:
        return raw
    return f"{raw[:4]}…{raw[-3:]}"


def _parse_nums(text: str) -> list[int]:
    """'1,3 5-8' -> [1, 3, 5, 6, 7, 8]. Ignores junk."""
    nums: list[int] = []
    for tok in re.findall(r"\d+\s*(?:-\s*\d+)?", text or ""):
        if "-" in tok:
            a, b = tok.split("-")
            lo, hi = int(a), int(b)
            if lo <= hi and hi - lo <= 50:
                nums.extend(range(lo, hi + 1))
        else:
            nums.append(int(tok))
    return nums


# ---------------------------------------------------------------------------
# panel state
# ---------------------------------------------------------------------------

class SeshPanel:
    """State holder for one /sesh picker embed."""

    def __init__(self) -> None:
        self.selected: set[str] = set()
        self.reveal = False
        self.projects: list[dict[str, Any]] = []
        self.discovered = False

    def refresh_projects(self) -> None:
        self.projects = _scan_projects()
        self.discovered = True
        self.selected &= {p["name"] for p in self.projects}

    def modal_ranges(self) -> list[tuple]:
        """[(1,15), (16,30), (31,45)] clamped to project count; only non-empty ones."""
        total = len(self.projects)
        ranges = []
        for i in range(3):
            lo, hi = i * MODAL_SPAN + 1, (i + 1) * MODAL_SPAN
            if lo <= total:
                ranges.append((lo, min(hi, total)))
        return ranges or [(1, 1)]

    def name_of(self, num: int) -> Optional[str]:
        if 1 <= num <= len(self.projects):
            return self.projects[num - 1]["name"]
        return None

    def status_fields(self) -> list[tuple]:
        settings = _load_settings()
        return [
            ("Selected", f"{len(self.selected)} project(s)"),
            ("Push", "on ✅" if settings.get("push", True) else "OFF ⚠️"),
            ("Max tasks each", str(settings.get("max_tasks_per_project", 999))),
            ("Model", (settings.get("worker_default") or "default").split(":")[-1]),
            ("Fallbacks", ", ".join(settings.get("worker_fallbacks") or []) or "none"),
        ]

    def build_embed(self) -> discord.Embed:
        lines: list[str] = []
        for i, proj in enumerate(self.projects, start=1):
            name = _mask(proj["name"], self.reveal)
            n = proj.get("open_tasks", 0)
            bits = [f"`{i:>2}` {'✅' if proj['name'] in self.selected else '☐'} **{name}** — {n} open"]
            bits.append(_age_label(proj.get("last_commit_days")))
            if proj.get("dirty"):
                bits.append("⚠️ uncommitted")
            if not proj.get("is_git"):
                bits.append("no git")
            lines.append(" · ".join(bits))
        if not lines:
            lines = ["(no projects found)"]
        embed = discord.Embed(
            title="🌙 /sesh — ALL projects",
            description="\n".join(lines)[:4000],
            colour=0x5865F2,
        )
        for label, value in self.status_fields():
            embed.add_field(name=label, value=value or "—", inline=True)
        embed.set_footer(text=f"{len(self.projects)} projects · tick via the Projects modals "
                              "· Settings/Model/Fallbacks open modals · runs always commit+push")
        return embed

    def chosen_arg(self) -> str:
        return ";".join(sorted(self.selected))


# ---------------------------------------------------------------------------
# modals
# ---------------------------------------------------------------------------

class ProjectModal(discord.ui.Modal):
    """Tick/untick projects by number. Covers MODAL_SPAN projects per modal."""

    tick = discord.ui.TextInput(label="Tick numbers", placeholder="e.g. 2, 5, 9-12",
                                required=False, max_length=200)
    untick = discord.ui.TextInput(label="Untick numbers", placeholder="e.g. 3, 7",
                                  required=False, max_length=200)

    def __init__(self, panel: SeshPanel, lo: int, hi: int) -> None:
        super().__init__(title=f"Projects {lo}-{hi}")
        self.panel = panel
        self.lo, self.hi = lo, hi
        self.tick.label = f"Tick {lo}-{hi}"
        self.untick.label = f"Untick {lo}-{hi}"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        panel = self.panel
        for num in _parse_nums(self.tick.value):
            name = panel.name_of(num)
            if name:
                panel.selected.add(name)
        for num in _parse_nums(self.untick.value):
            name = panel.name_of(num)
            if name:
                panel.selected.discard(name)
        await interaction.response.edit_message(embed=panel.build_embed(),
                                                view=_build_view(panel))


class SettingsModal(discord.ui.Modal):
    push = discord.ui.TextInput(label="Push on/off", placeholder="on", max_length=3, required=False)
    time_cap = discord.ui.TextInput(label="Time cap (minutes)", max_length=5, required=False)
    cost_cap = discord.ui.TextInput(label="Cost cap (USD)", max_length=6, required=False)
    max_tasks = discord.ui.TextInput(label="Max tasks per project", max_length=4, required=False)
    max_parallel = discord.ui.TextInput(label="Max parallel workers", max_length=2, required=False)

    def __init__(self, panel: SeshPanel) -> None:
        super().__init__(title="⚙️ sesh settings")
        self.panel = panel
        s = _load_settings()
        self.push.default = "on" if s.get("push", True) else "off"
        self.time_cap.default = str(s.get("run_time_cap_min", 360))
        self.cost_cap.default = str(s.get("run_cost_cap_usd", 3))
        self.max_tasks.default = str(s.get("max_tasks_per_project", 999))
        self.max_parallel.default = str(s.get("max_parallel_workers", 2))

    async def on_submit(self, interaction: discord.Interaction) -> None:
        s = _load_settings()
        if self.push.value.strip().lower() in ("on", "off"):
            s["push"] = self.push.value.strip().lower() == "on"
        for field, key in ((self.time_cap, "run_time_cap_min"),
                           (self.max_tasks, "max_tasks_per_project"),
                           (self.max_parallel, "max_parallel_workers")):
            if field.value.strip().isdigit():
                s[key] = int(field.value.strip())
        if self.cost_cap.value.strip():
            try:
                s["run_cost_cap_usd"] = float(self.cost_cap.value.strip().lstrip("$"))
            except ValueError:
                pass
        _save_settings(s)
        await interaction.response.edit_message(embed=self.panel.build_embed(),
                                                view=_build_view(self.panel))


class ModelModal(discord.ui.Modal):
    model = discord.ui.TextInput(label="Worker model (exact spec)",
                                 placeholder="hermes:openrouter:nvidia/nemotron-3-ultra-550b-a55b:free",
                                 max_length=200)

    def __init__(self, panel: SeshPanel) -> None:
        super().__init__(title="🤖 worker model")
        self.panel = panel
        self.model.default = _load_settings().get("worker_default", "")

    async def on_submit(self, interaction: discord.Interaction) -> None:
        val = self.model.value.strip()
        s = _load_settings()
        if val:
            s["worker_default"] = val
        _save_settings(s)
        await interaction.response.edit_message(embed=self.panel.build_embed(),
                                                view=_build_view(self.panel))


class FallbacksModal(discord.ui.Modal):
    fallbacks = discord.ui.TextInput(label="Fallback models (comma separated)",
                                     placeholder="hermes:nous:reflection/beam-1.0-501b-a23b:free, ...",
                                     max_length=600, style=discord.TextStyle.paragraph)

    def __init__(self, panel: SeshPanel) -> None:
        super().__init__(title="🔁 fallback models")
        self.panel = panel
        self.fallbacks.default = ", ".join(_load_settings().get("worker_fallbacks") or [])

    async def on_submit(self, interaction: discord.Interaction) -> None:
        items = [x.strip() for x in self.fallbacks.value.replace(";", ",").split(",") if x.strip()]
        s = _load_settings()
        s["worker_fallbacks"] = items
        _save_settings(s)
        await interaction.response.edit_message(embed=self.panel.build_embed(),
                                                view=_build_view(self.panel))


# ---------------------------------------------------------------------------
# main view builder (always <= 5 rows)
# ---------------------------------------------------------------------------

def _build_view(panel: SeshPanel) -> discord.ui.View:
    assert discord is not None

    class SeshView(discord.ui.View):
        def __init__(self) -> None:
            super().__init__(timeout=None)
            for i, (lo, hi) in enumerate(panel.modal_ranges()):
                btn = discord.ui.Button(label=f"Projects {lo}–{hi}",
                                        style=discord.ButtonStyle.primary, row=0,
                                        custom_id=f"projmodal{i}")
                btn.callback = self._make_project_cb(lo, hi)
                self.add_item(btn)

        def _make_project_cb(self, lo: int, hi: int):
            async def cb(interaction: discord.Interaction) -> None:
                await interaction.response.send_modal(ProjectModal(panel, lo, hi))
            return cb

        @discord.ui.button(label="⚙️ Settings", style=discord.ButtonStyle.secondary, row=1)
        async def settings_btn(self, interaction: discord.Interaction, _btn) -> None:
            await interaction.response.send_modal(SettingsModal(panel))

        @discord.ui.button(label="🤖 Model", style=discord.ButtonStyle.secondary, row=1)
        async def model_btn(self, interaction: discord.Interaction, _btn) -> None:
            await interaction.response.send_modal(ModelModal(panel))

        @discord.ui.button(label="🔁 Fallbacks", style=discord.ButtonStyle.secondary, row=1)
        async def fallbacks_btn(self, interaction: discord.Interaction, _btn) -> None:
            await interaction.response.send_modal(FallbacksModal(panel))

        @discord.ui.button(label="Refresh", style=discord.ButtonStyle.secondary, row=2)
        async def refresh(self, interaction: discord.Interaction, _btn) -> None:
            await interaction.response.defer()
            await asyncio.get_event_loop().run_in_executor(None, panel.refresh_projects)
            await interaction.edit_original_response(embed=panel.build_embed(),
                                                     view=_build_view(panel))

        @discord.ui.button(label="Clear", style=discord.ButtonStyle.secondary, row=2)
        async def clear(self, interaction: discord.Interaction, _btn) -> None:
            panel.selected.clear()
            await interaction.response.edit_message(embed=panel.build_embed(), view=self)

        @discord.ui.button(label="Reveal names", style=discord.ButtonStyle.secondary, row=2)
        async def reveal(self, interaction: discord.Interaction, _btn) -> None:
            panel.reveal = not panel.reveal
            await interaction.response.edit_message(embed=panel.build_embed(), view=self)

        @discord.ui.button(label="Start run", style=discord.ButtonStyle.success, row=2)
        async def start(self, interaction: discord.Interaction, _btn) -> None:
            if not panel.selected:
                await interaction.response.send_message(
                    "Tick at least one project first (Projects modals).", ephemeral=True)
                return
            settings = _load_settings()
            if not settings.get("push", True):
                await interaction.response.send_message(
                    "⚠ Push is off in settings.json — turn it on before starting "
                    "(runs must commit + push).", ephemeral=True)
                return
            arg = panel.chosen_arg()
            stamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = f"{RUNS_DIR}/ui-{stamp}.log"
            cmd = [sys.executable, RUNNER_SCRIPT, "--projects", arg,
                   "--timeout-min", str(min(settings.get("run_time_cap_min", 360), 360))]
            try:
                os.makedirs(RUNS_DIR, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as logf:  # noqa: ASYNC230
                    subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,  # noqa: ASYNC220
                                     stdin=subprocess.DEVNULL)
            except Exception as exc:
                await interaction.response.send_message(f"❌ Failed to start runner: {exc}",
                                                        ephemeral=True)
                return
            embed = discord.Embed(
                title="🚀 sesh run started",
                description=f"Projects: {arg}\nLog: `{log_path}`",
                colour=0x57F287,
            )
            embed.set_footer(text="Workers commit + push per project; /nitesesh for status.")
            await interaction.response.edit_message(embed=embed, view=None)

        @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, row=2)
        async def cancel(self, interaction: discord.Interaction, _btn) -> None:
            await interaction.response.edit_message(
                embed=discord.Embed(title="/sesh — cancelled", colour=0xED4245), view=None)

    return SeshView()


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def register(adapter: Any, tree: Any) -> None:
    """Register /sesh and /nitesesh on the Discord command tree.

    Called from DiscordAdapter._register_slash_commands. Must never raise.
    """
    if discord is None:
        raise RuntimeError("discord.py not available")

    async def _sesh_handler(interaction: discord.Interaction) -> None:
        panel = SeshPanel()
        await interaction.response.defer(ephemeral=False, thinking=True)
        await asyncio.get_event_loop().run_in_executor(None, panel.refresh_projects)
        await interaction.edit_original_response(embed=panel.build_embed(),
                                                 view=_build_view(panel))

    _sesh_handler.__name__ = "sesh_panel"
    try:
        if tree.get_command("sesh") is not None:
            tree.remove_command("sesh")
    except Exception:
        pass
    tree.command(name="sesh",
                 description="Pick projects and start sesh workers (interactive panel)")(
        _sesh_handler)

    async def _nitesesh_handler(interaction: discord.Interaction) -> None:
        s = _load_settings()
        runs_dir = Path(RUNS_DIR)
        recent = []
        if runs_dir.is_dir():
            logs = sorted(runs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            recent = [f"`{p.name}`" for p in logs]
        embed = discord.Embed(
            title="/nitesesh — nightly status",
            description=(
                f"Schedule: **{s.get('schedule', 'manual')}**\n"
                f"Projects saved: {len([p for p in s.get('projects', []) if not str(p).startswith('##')])}\n"
                f"Push: {'on' if s.get('push') else 'off'} · "
                f"caps: {s.get('run_time_cap_min')} min / ${s.get('run_cost_cap_usd')}\n"
                f"Model: `{s.get('worker_default', '(unset)')}`\n"
                f"Fallbacks: {', '.join(s.get('worker_fallbacks') or []) or 'none'}\n"
                f"Max parallel: {s.get('max_parallel_workers', 2)} · "
                f"Max tasks/project: {s.get('max_tasks_per_project', 999)}\n"
                + (f"\nRecent run logs: {', '.join(recent)}" if recent else "")
            ),
            colour=0xFEE75C,
        )
        embed.set_footer(text="Runs must always commit + push (user rule 2026-11-23).")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    _nitesesh_handler.__name__ = "nitesesh_panel"
    tree.command(name="nitesesh",
                 description="Show the nitesesh nightly schedule and settings")(
        _nitesesh_handler)

    logger.info("nitesesh UI registered: /sesh panel + /nitesesh status")
