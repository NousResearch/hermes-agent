"""The /sesh + /nitesesh interactive Discord panel (Components V2 rebuild).

Lives as a standalone file on purpose: Hermes updates never touch it. The only
integration point is a try/except hook in DiscordAdapter._register_slash_commands
(see Z:/HERMES/.hermes/nitesesh/REAPPLY.md). Restore both after `hermes update`.

Panel behaviour (user spec 2026-11-24, rev 2):
  /sesh -> ONE Components-V2 message. NO buttons for selection:
    - default model = dropdown on the message (Nvidia Free / OpenCode Free / Nous)
    - fallbacks = multi-select dropdown on the message (max 3; sesh ignores
      Hermes' built-in fallbacks — these 3 are the only ones the runner uses)
    - projects = multi-select dropdowns on the message (25 per select)
    Buttons ONLY: Start run, Refresh, Reveal names, Prep tasks.
  Prep sweeps every project's todo.md and counts open tasks (the discovery
  script undercounts, showing 0 for projects that have open work).
  /nitesesh -> status embed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import discord
except ImportError:  # pragma: no cover - adapter only loads this with discord present
    discord = None

logger = logging.getLogger(__name__)

SETTINGS_PATH = "Z:/HERMES/.hermes/nitesesh/settings.json"
DISCOVER_SCRIPT = "Z:/HERMES/.hermes/scripts/discover-todo-projects.py"
RUNNER_SCRIPT = "Z:/HERMES/.hermes/scripts/nitesesh-run.py"
RUNS_DIR = "Z:/HERMES/.hermes/nitesesh/runs"
TASKCOUNTS_PATH = "Z:/HERMES/.hermes/nitesesh/taskcounts.json"
SELECT_SPAN = 25  # Discord hard cap on options per select


# ---------------------------------------------------------------------------
# settings + git helpers
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


def _load_taskcounts() -> dict[str, int]:
    try:
        with open(TASKCOUNTS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_taskcounts(counts: dict[str, int]) -> None:
    with open(TASKCOUNTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(counts, fh, indent=2)


def _git_out(path: str, *args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", path, *args], check=False, capture_output=True,
                           text=True, timeout=15)
        return r.stdout.strip()
    except Exception:
        return ""


def _age_label(days_ago: int | None) -> str:
    if days_ago is None:
        return "no commits"
    if days_ago <= 0:
        return "just now"
    if days_ago == 1:
        return "1d ago"
    return f"{days_ago}d ago"


# ---------------------------------------------------------------------------
# model choices — real free lists (OpenCode free fetched from the Zen API
# 2026-11-24: price 0 across the board)
# ---------------------------------------------------------------------------

MODEL_CHOICES: list[tuple[str, str, str]] = [
    # (label, spec, provider group)
    ("Nemotron 3 Ultra 550B (free)", "hermes:openrouter:nvidia/nemotron-3-ultra-550b-a55b:free", "Nvidia Free"),
    ("Beam Reflection 501B (free)", "hermes:nous:reflection/beam-1.0-501b-a23b:free", "Nous"),
    ("GLM 5.3 Flash", "hermes:z-ai:glm-5.3-flash", "Nvidia Free"),
    ("DeepSeek V4 Pro", "hermes:openrouter:deepseek/deepseek-v4-pro", "Nvidia Free"),
    ("Inkling (Thinking Machines, free)", "hermes:openrouter:thinkingmachines/inkling:free", "Nvidia Free"),
    # OpenCode free tier (verified zero-cost)
    ("Big Pickle", "hermes:opencode-zen:big-pickle", "OpenCode Free"),
    ("MiMo-V2.6 Flash Free", "hermes:opencode-zen:mimo-v2.6-flash-free", "OpenCode Free"),
    ("DeepSeek V4.1 Flash Free", "hermes:opencode-zen:deepseek-v4.1-flash-free", "OpenCode Free"),
    ("Iceman Free", "hermes:opencode-zen:iceman-free", "OpenCode Free"),
    ("Beam Free", "hermes:opencode-zen:beam-free", "OpenCode Free"),
    ("Flint 40B Free", "hermes:opencode-zen:flint-40b-free", "OpenCode Free"),
    ("Nemotron 3.5 Super VL Free", "hermes:opencode-zen:nemotron-3.5-super-vl-free", "OpenCode Free"),
    ("Nemotron 3.5 Lightning Free", "hermes:opencode-zen:nemotron-3.5-lightning-free", "OpenCode Free"),
    ("Jev 1.13 Free", "hermes:opencode-zen:jev-1.13-free", "OpenCode Free"),
    ("Hy4 Flash Free", "hermes:opencode-zen:hy4-flash-free", "OpenCode Free"),
    ("Tidepool Alpha Free", "hermes:opencode-zen:tidepool-alpha-free", "OpenCode Free"),
]


# ---------------------------------------------------------------------------
# project enumeration
# ---------------------------------------------------------------------------

def _find_repo_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    roots = cfg.get("roots") or ["Z:/github"]
    extra = cfg.get("extra_projects") or []
    saved_names = [n for n in (cfg.get("projects") or []) if not str(n).startswith("##")]
    skip_names = {"node_modules", "__pycache__", "dist", "build", "release",
                  ".git", ".venv", "venv", "build-target", "target", "assets"}

    found: dict[str, Path] = {}
    for r in roots:
        root = Path(r)
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, _ in os.walk(root):
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

    for path in extra:
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
    return found


def _prep_tasks(projects: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Sweep every project's todo.md and count unchecked tasks.

    The discovery script undercounts (many repos report 0 despite open work),
    so this reads the todo files directly. Results land in taskcounts.json and
    are used by the panel from then on.
    """
    if projects is None:
        projects = _scan_projects()
    counts: dict[str, int] = dict(_load_taskcounts())
    for proj in projects:
        base = Path(proj["path"])
        cands = [base / "todo.md", base / "TODO.md", base / "Todo.md"]
        for sub in ("docs", ".hermes", "plans"):
            cands += [base / sub / "todo.md", base / sub / "TODO.md"]
        todo = next((c for c in cands if c.is_file()), None)
        n = 0
        if todo is not None:
            try:
                for line in todo.read_text(encoding="utf-8", errors="replace").splitlines():
                    stripped = line.strip().lower()
                    if stripped.startswith(("- [ ]", "* [ ]", "+ [ ]")):
                        n += 1
            except Exception:
                pass
        counts[str(base).lower()] = n
    _save_taskcounts(counts)
    return counts


def _scan_projects() -> list[dict[str, Any]]:
    cfg = _load_settings()
    found = _find_repo_dirs(cfg)
    prep_counts = _load_taskcounts()

    open_map: dict[str, int] = {}
    try:
        out = subprocess.run([sys.executable, DISCOVER_SCRIPT],
                             capture_output=True, text=True, timeout=180, check=True)
        for item in json.loads(out.stdout).get("projects", []):
            open_map[str(Path(item["path"])).lower()] = int(item.get("open_tasks", 0))
    except Exception as exc:
        logger.warning("nitesesh_ui: discovery lookup failed: %s", exc)

    def merge_counts(path: str) -> int:
        # prep sweep wins when it found more than discovery did
        return max(open_map.get(path, 0), prep_counts.get(path, 0))

    out: list[dict[str, Any]] = []
    for _key, proj in sorted(found.items()):
        is_git = (proj / ".git").exists()
        dirty, days = False, None
        if is_git:
            dirty = bool(_git_out(str(proj), "status", "--porcelain", "--untracked-files=no"))
            ts = _git_out(str(proj), "log", "-1", "--format=%ct")
            if ts.isdigit():
                days = int(time.time() - int(ts)) // 86400
        out.append({"name": proj.name, "path": str(proj), "open_tasks": merge_counts(str(proj).lower()),
                    "is_git": is_git, "dirty": dirty, "last_commit_days": days})

    def sort_key(item):
        if not item["is_git"]:
            return (2, 0)
        if item["last_commit_days"] is None:
            return (1, 0)
        return (0, item["last_commit_days"])
    out.sort(key=sort_key)

    # dedupe case/name variants, keep the most recently committed entry
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


# ---------------------------------------------------------------------------
# panel state
# ---------------------------------------------------------------------------

class SeshPanel:
    def __init__(self) -> None:
        self.selected: set[str] = set()
        self.reveal = False
        self.projects: list[dict[str, Any]] = []

    def refresh_projects(self) -> None:
        self.projects = _scan_projects()
        self.selected &= {p["name"] for p in self.projects}

    def name_of(self, num: int) -> str | None:
        if 1 <= num <= len(self.projects):
            return self.projects[num - 1]["name"]
        return None

    def project_select_count(self) -> int:
        return max(1, math.ceil(len(self.projects) / SELECT_SPAN))

    def header_text(self) -> str:
        s = _load_settings()
        model = (s.get("worker_default") or "default").split(":")[-1]
        fbs = ", ".join((s.get("worker_fallbacks") or [])[:3]) or "none"
        return (f"## 🌙 /sesh — pick up work\n"
                f"**Selected:** {len(self.selected)} · **Push:** "
                f"{'on ✅' if s.get('push', True) else 'OFF ⚠️'} · "
                f"**Max tasks each:** {s.get('max_tasks_per_project', 999)} · "
                f"**Model:** {model} · **Fallbacks (max 3):** {fbs}\n"
                f"-# Pick projects with the dropdowns below · runs always commit+push · "
                f"Prep re-counts todo.md tasks · {len(self.projects)} projects")

    def project_list_text(self) -> str:
        lines = []
        for i, proj in enumerate(self.projects, start=1):
            name = _mask(proj["name"], self.reveal)
            bits = [f"`{i:>2}` {'✅' if proj['name'] in self.selected else '☐'} **{name}** — "
                    f"{proj.get('open_tasks', 0)} open",
                    _age_label(proj.get("last_commit_days"))]
            if proj.get("dirty"):
                bits.append("⚠️ uncommitted")
            if not proj.get("is_git"):
                bits.append("no git")
            lines.append(" · ".join(bits))
        return "\n".join(lines)[:3800] if lines else "(no projects found)"

    def chosen_arg(self) -> str:
        return ";".join(sorted(self.selected))


# ---------------------------------------------------------------------------
# Components V2 panel (selects on the message, buttons only for actions)
# ---------------------------------------------------------------------------

def _proj_option(idx: int, proj: dict[str, Any], panel: SeshPanel) -> discord.SelectOption:
    name = _mask(proj["name"], panel.reveal)
    desc = f"{proj.get('open_tasks', 0)} open · {_age_label(proj.get('last_commit_days'))}"
    if proj.get("dirty"):
        desc += " · ⚠️ uncommitted"
    return discord.SelectOption(label=name[:100], value=proj["name"],
                                description=desc[:100], default=proj["name"] in panel.selected)


def _build_panel_view(panel: SeshPanel) -> discord.ui.LayoutView:
    assert discord is not None
    settings = _load_settings()

    class SeshLayout(discord.ui.LayoutView):
        def __init__(self) -> None:
            super().__init__(timeout=None)
            self.add_item(discord.ui.TextDisplay(panel.header_text()))
            self.add_item(discord.ui.TextDisplay(panel.project_list_text()))
            self.add_item(discord.ui.Separator())

            # project multi-selects (25 options each)
            for pi in range(panel.project_select_count()):
                chunk = panel.projects[pi * SELECT_SPAN:(pi + 1) * SELECT_SPAN]
                lo, hi = pi * SELECT_SPAN + 1, pi * SELECT_SPAN + len(chunk)
                sel = discord.ui.Select(placeholder=f"Projects {lo}–{hi} (tick to run)",
                                        min_values=0, max_values=len(chunk),
                                        options=[_proj_option(lo + j, p, panel) for j, p in enumerate(chunk)],
                                        custom_id=f"sesh-proj-{pi}")
                sel.callback = self._make_proj_cb(sel)
                self.add_item(sel)

            self.add_item(discord.ui.Separator())

            # default model dropdown
            current_model = settings.get("worker_default", "")
            model_sel = discord.ui.Select(
                placeholder="Default model — Nvidia Free / OpenCode Free / Nous",
                min_values=1, max_values=1,
                options=[discord.SelectOption(label=label[:100], value=spec,
                                              description=f"{group}"[:100],
                                              default=(spec == current_model))
                         for label, spec, group in MODEL_CHOICES],
                custom_id="sesh-model")
            model_sel.callback = self._make_model_cb(model_sel)
            self.add_item(model_sel)

            # fallbacks multi-select (max 3)
            current_fbs = list(settings.get("worker_fallbacks") or [])[:3]
            fb_sel = discord.ui.Select(
                placeholder="Fallbacks — pick up to 3 (sesh ignores Hermes default fallbacks)",
                min_values=0, max_values=3,
                options=[discord.SelectOption(label=label[:100], value=spec,
                                              description=f"{group}"[:100],
                                              default=(spec in current_fbs))
                         for label, spec, group in MODEL_CHOICES],
                custom_id="sesh-fallbacks")
            fb_sel.callback = self._make_fallback_cb(fb_sel)
            self.add_item(fb_sel)

            row = discord.ui.ActionRow()
            start_btn = discord.ui.Button(label="Start run", style=discord.ButtonStyle.success,
                                          custom_id="sesh-start")
            start_btn.callback = self._start_cb
            row.add_item(start_btn)
            refresh_btn = discord.ui.Button(label="Refresh", style=discord.ButtonStyle.secondary,
                                            custom_id="sesh-refresh")
            refresh_btn.callback = self._refresh_cb
            row.add_item(refresh_btn)
            reveal_btn = discord.ui.Button(label="Reveal names", style=discord.ButtonStyle.secondary,
                                           custom_id="sesh-reveal")
            reveal_btn.callback = self._reveal_cb
            row.add_item(reveal_btn)
            prep_btn = discord.ui.Button(label="Prep tasks", style=discord.ButtonStyle.primary,
                                         custom_id="sesh-prep")
            prep_btn.callback = self._prep_cb
            row.add_item(prep_btn)
            self.add_item(row)

        # ---- callbacks -------------------------------------------------
        def _make_proj_cb(self, sel):
            async def cb(interaction: discord.Interaction) -> None:
                names: set[str] = set()
                for child in self.children:
                    if isinstance(child, discord.ui.Select) and (child.custom_id or "").startswith("sesh-proj"):
                        names.update(child.values)
                panel.selected = names
                await self._redraw(interaction)
            return cb

        def _make_model_cb(self, sel):
            async def cb(interaction: discord.Interaction) -> None:
                s = _load_settings()
                s["worker_default"] = sel.values[0]
                _save_settings(s)
                await self._redraw(interaction)
            return cb

        def _make_fallback_cb(self, sel):
            async def cb(interaction: discord.Interaction) -> None:
                s = _load_settings()
                s["worker_fallbacks"] = list(sel.values)[:3]
                _save_settings(s)
                await self._redraw(interaction)
            return cb

        async def _redraw(self, interaction: discord.Interaction) -> None:
            fresh = _build_panel_view(panel)
            await interaction.response.edit_message(view=fresh)

        async def _refresh_cb(self, interaction: discord.Interaction) -> None:
            await interaction.response.defer()
            await asyncio.get_event_loop().run_in_executor(None, panel.refresh_projects)
            await interaction.edit_original_response(view=_build_panel_view(panel))

        async def _reveal_cb(self, interaction: discord.Interaction) -> None:
            panel.reveal = not panel.reveal
            await self._redraw(interaction)

        async def _prep_cb(self, interaction: discord.Interaction) -> None:
            await interaction.response.defer()
            def sweep():
                projs = panel.projects or _scan_projects()
                _prep_tasks(projs)
                panel.refresh_projects()
            await asyncio.get_event_loop().run_in_executor(None, sweep)
            done = _build_panel_view(panel)
            done.add_item(discord.ui.TextDisplay(
                "-# ✅ Prep sweep done — todo.md open-task counts updated for all projects."))
            await interaction.edit_original_response(view=done)

        async def _start_cb(self, interaction: discord.Interaction) -> None:
            if not panel.selected:
                await interaction.response.send_message(
                    "Tick at least one project in the project dropdowns first.", ephemeral=True)
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
            done = _build_panel_view(panel)
            done.add_item(discord.ui.TextDisplay(
                f"## 🚀 sesh run started\nProjects: {arg}\n-# Log: `{log_path}` · "
                f"workers commit + push per project"))
            await interaction.response.edit_message(view=done)

    return SeshLayout()


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def register(adapter: Any, tree: Any) -> None:
    """Register /sesh and /nitesesh on the Discord command tree. Must never raise."""
    if discord is None:
        raise RuntimeError("discord.py not available")

    async def _sesh_handler(interaction: discord.Interaction) -> None:
        panel = SeshPanel()
        await interaction.response.defer(ephemeral=False, thinking=True)
        await asyncio.get_event_loop().run_in_executor(None, panel.refresh_projects)
        await interaction.edit_original_response(view=_build_panel_view(panel))

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

    logger.info("nitesesh UI registered: /sesh panel (Components V2) + /nitesesh status")
