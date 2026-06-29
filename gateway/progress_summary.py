"""Concise human-facing progress summaries for gateway chat surfaces.

The raw tool-progress stream is useful in a terminal, but noisy on mobile chat
clients: long shell commands become half-visible code blocks and repeated
``process wait`` calls are poor signal.  ``ToolProgressSummary`` consumes the
same display-only lifecycle events and renders one editable operational status
bubble: current stage, safe action detail, counters, and delegated agent/profile
status when it can be inferred.

No tool output is included here.  The summary intentionally avoids raw shell
commands and only extracts low-risk identifiers such as Hermes profile names,
Kanban task IDs, file paths, and tool names.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import re
import shlex
from typing import Any, Optional


_SECRETISH_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_\-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|xox[baprs]-[A-Za-z0-9\-]{8,}|bearer\s+[A-Za-z0-9._\-]{8,})"
)
_KANBAN_TASK_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]{0,8}_[A-Za-z0-9]{6,}|t_[A-Za-z0-9]{6,})\b")
_PROCESS_WAIT_RE = re.compile(r"\bwait\s+(proc_[A-Za-z0-9]+)(?:\s+(\d+)s?)?", re.IGNORECASE)


class ToolProgressSummary:
    """Stateful renderer for ``display.tool_progress: summary``.

    ``update()`` returns the full replacement text for the editable progress
    bubble.  Callers should replace the previous progress bubble with this text
    rather than append it as a new line.
    """

    def __init__(self) -> None:
        self.started = 0
        self.completed = 0
        self.failed = 0
        self.subagent_tool_count = 0
        self.stage = "Préparation"
        self.detail = "Démarrage du traitement"
        self.tools: Counter[str] = Counter()
        self.recent_tools: list[str] = []
        self.profiles: "OrderedDict[str, dict[str, str]]" = OrderedDict()
        self.subagents: "OrderedDict[str, dict[str, str]]" = OrderedDict()
        self.moa_refs_done = 0
        self.moa_refs_total: Optional[int] = None

    def update(
        self,
        event_type: str,
        tool_name: Optional[str] = None,
        preview: Optional[str] = None,
        args: Any = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Ingest one gateway progress event and return a replacement summary."""

        event = str(event_type or "")
        if event == "tool.started":
            self._tool_started(str(tool_name or "outil"), preview, args)
            return self.render()
        if event == "tool.completed":
            self._tool_completed(str(tool_name or "outil"), kwargs)
            return self.render()
        if event.startswith("subagent") or event == "subagent_progress":
            self._subagent_event(event, tool_name, preview, args, kwargs)
            return self.render()
        if event.startswith("moa."):
            self._moa_event(event, tool_name, preview, kwargs)
            return self.render()
        return None

    def _tool_started(self, name: str, preview: Optional[str], args: Any) -> None:
        self.started += 1
        self.tools[name] += 1
        self._remember_tool(name)
        self.stage, self.detail = self._describe_tool(name, preview, args)

    def _tool_completed(self, name: str, kwargs: dict[str, Any]) -> None:
        self.completed += 1
        if bool(kwargs.get("is_error")):
            self.failed += 1
            self.stage = "Point de contrôle"
            self.detail = f"{_safe_text(name)} a retourné une erreur"
            return
        duration = _format_duration(kwargs.get("duration"))
        if duration:
            self.detail = f"{_safe_text(name)} terminé en {duration}"
        else:
            self.detail = f"{_safe_text(name)} terminé"

    def _subagent_event(
        self,
        event: str,
        tool_name: Optional[str],
        preview: Optional[str],
        args: Any,
        kwargs: dict[str, Any],
    ) -> None:
        raw_ident = (
            kwargs.get("subagent_id")
            or kwargs.get("child_session_id")
        )
        if raw_ident is None and "task_index" in kwargs:
            raw_ident = kwargs.get("task_index")
        ident = str(raw_ident if raw_ident is not None else f"agent-{len(self.subagents) + 1}")
        rec = self.subagents.setdefault(
            ident,
            {
                "label": self._subagent_label(kwargs),
                "goal": _shorten(kwargs.get("goal") or preview or "", 70),
                "status": "actif",
                "last": "initialisation",
            },
        )
        if kwargs.get("goal") and not rec.get("goal"):
            rec["goal"] = _shorten(kwargs.get("goal"), 70)
        if kwargs.get("model") and "model" not in rec:
            rec["model"] = _shorten(kwargs.get("model"), 42)

        if event in {"subagent.spawn_requested", "subagent.start"}:
            rec["status"] = "actif"
            rec["last"] = "démarrage"
            self.stage = "Délégation"
            self.detail = f"Agent {rec['label']} lancé"
        elif event in {"subagent.complete", "subagent.completed"}:
            rec["status"] = "terminé"
            rec["last"] = "résultat reçu"
            self.stage = "Délégation"
            self.detail = f"Agent {rec['label']} terminé"
        elif event in {"subagent.tool", "delegate.task_tool_started"}:
            self.subagent_tool_count += 1
            rec["status"] = "actif"
            rec["last"] = _safe_text(tool_name or "outil")
            self.stage = "Délégation"
            self.detail = f"Agent {rec['label']} utilise {rec['last']}"
        elif event in {"subagent.progress", "subagent_progress"}:
            summary = _clean_preview(preview or tool_name or "")
            if summary:
                rec["last"] = _shorten(summary, 70)
                self.stage = "Délégation"
                self.detail = rec["last"]
        elif event in {"subagent.text", "subagent.thinking"}:
            # Do not mirror child free text into the compact operational bubble;
            # interim assistant messages already carry deliberate user-facing text.
            self.stage = "Délégation"
            self.detail = f"Agent {rec['label']} rédige ou analyse"

    def _subagent_label(self, kwargs: dict[str, Any]) -> str:
        raw_index = kwargs.get("task_index")
        if raw_index is not None:
            try:
                idx = int(raw_index) + 1
                base = f"#{idx}"
            except Exception:
                base = f"#{len(self.subagents) + 1}"
        else:
            base = f"#{len(self.subagents) + 1}"
        model = kwargs.get("model")
        if model:
            return f"{base} {_shorten(model, 28)}"
        return base

    def _moa_event(
        self,
        event: str,
        tool_name: Optional[str],
        preview: Optional[str],
        kwargs: dict[str, Any],
    ) -> None:
        if event == "moa.reference":
            self.moa_refs_done = max(self.moa_refs_done, _safe_int(kwargs.get("moa_index"), -1) + 1)
            total = _safe_int(kwargs.get("moa_count"), 0)
            if total > 0:
                self.moa_refs_total = total
            label = _shorten(tool_name or "référence", 40)
            suffix = f" {self.moa_refs_done}/{self.moa_refs_total}" if self.moa_refs_total else ""
            self.stage = "Consultation MoA"
            self.detail = f"Réponse de référence{suffix}: {label}"
        elif event == "moa.aggregating":
            total = _safe_int(kwargs.get("moa_ref_count"), 0) or self.moa_refs_total
            self.stage = "Synthèse MoA"
            self.detail = f"Agrégation de {total} réponse(s)" if total else "Agrégation des réponses"

    def _describe_tool(self, name: str, preview: Optional[str], args: Any) -> tuple[str, str]:
        argd = args if isinstance(args, dict) else {}
        if name in {"read_file", "browser_snapshot"}:
            path = argd.get("path") or preview
            return "Lecture contexte", f"Lecture {_shorten(path or 'du contexte', 90)}"
        if name in {"search_files", "session_search"}:
            query = argd.get("pattern") or argd.get("query") or preview
            return "Recherche contexte", f"Recherche {_shorten(query or 'ciblée', 90)}"
        if name in {"skill_view", "skills_list"}:
            skill = argd.get("name") or preview
            return "Procédure", f"Chargement compétence {_shorten(skill or '', 70)}".rstrip()
        if name in {"todo"}:
            items = argd.get("todos")
            if isinstance(items, list):
                return "Planification", f"Plan de travail mis à jour ({len(items)} tâche(s))"
            return "Planification", "Lecture du plan de travail"
        if name in {"terminal"}:
            return self._describe_terminal(argd.get("command") or preview or "")
        if name in {"execute_code"}:
            return "Analyse script", "Exécution d’un script Python de contrôle"
        if name in {"patch", "write_file"}:
            path = argd.get("path") or preview
            return "Modification fichiers", f"Mise à jour {_shorten(path or 'fichier', 90)}"
        if name in {"delegate_task"}:
            count = len(argd.get("tasks") or []) or 1
            return "Délégation", f"Lancement de {count} agent(s) délégué(s)"
        if name in {"cronjob"}:
            action = argd.get("action") or preview
            return "Planification", f"Cron: {_shorten(action or 'opération', 60)}"
        if name.startswith("kanban"):
            task = argd.get("task_id") or argd.get("id") or _extract_kanban_task(str(preview or ""))
            return "Kanban", f"{name}" + (f" sur {task}" if task else "")
        if name in {"process", "background_process"}:
            return self._describe_process(preview or argd.get("action") or "")
        if name.startswith("browser"):
            return "Navigation web", _shorten(preview or name, 90)
        if name.startswith("image") or name.startswith("vision"):
            return "Analyse média", _shorten(preview or name, 90)
        clean = _clean_preview(preview or "")
        return "Travail en cours", f"{name}" + (f" — {_shorten(clean, 90)}" if clean else "")

    def _describe_terminal(self, command: str) -> tuple[str, str]:
        command = str(command or "")
        profile, task_id = _extract_hermes_profile_and_task(command)
        if profile:
            info = self.profiles.setdefault(profile, {"status": "actif"})
            if task_id:
                info["task"] = task_id
            detail = f"Profil {profile}"
            if task_id:
                detail += f" — tâche Kanban {task_id}"
            return "Délégation profil", f"Lancement {detail}"
        if _PROCESS_WAIT_RE.search(command):
            return self._describe_process(command)
        return "Commande système", _summarize_command_shape(command)

    def _describe_process(self, text: str) -> tuple[str, str]:
        match = _PROCESS_WAIT_RE.search(str(text or ""))
        if match:
            proc_id = match.group(1)
            waited = match.group(2)
            detail = f"Attente processus {proc_id}"
            if waited:
                detail += f" ({waited}s)"
            return "Suivi processus", detail
        clean = _clean_preview(text)
        return "Suivi processus", _shorten(clean or "Contrôle d’un processus en arrière-plan", 90)

    def _remember_tool(self, name: str) -> None:
        self.recent_tools.append(name)
        if len(self.recent_tools) > 5:
            self.recent_tools = self.recent_tools[-5:]

    def render(self) -> str:
        lines = ["📊 Suivi du travail"]
        lines.append(f"Étape: {self.stage}")
        if self.detail:
            lines.append(f"Action: {self.detail}")

        counters = []
        if self.started:
            counters.append(f"outils {self.completed}/{self.started} terminés")
        if self.failed:
            counters.append(f"erreurs {self.failed}")
        if self.subagent_tool_count:
            counters.append(f"outils agents {self.subagent_tool_count}")
        if counters:
            lines.append("Compteurs: " + " · ".join(counters))

        if self.profiles:
            rendered = []
            for profile, info in list(self.profiles.items())[-3:]:
                task = info.get("task")
                rendered.append(f"{profile}" + (f"({task})" if task else ""))
            lines.append("Profils: " + ", ".join(rendered))

        if self.subagents:
            active = sum(1 for rec in self.subagents.values() if rec.get("status") != "terminé")
            done = sum(1 for rec in self.subagents.values() if rec.get("status") == "terminé")
            lines.append(f"Agents: {active} actif(s), {done} terminé(s)")
            for rec in list(self.subagents.values())[-3:]:
                tail = rec.get("last") or rec.get("goal") or rec.get("status") or ""
                lines.append(f"- {rec.get('label', '#?')}: {_shorten(tail, 80)}")

        if self.tools:
            common = ", ".join(
                f"{name}×{count}" for name, count in self.tools.most_common(4)
            )
            lines.append("Outils: " + common)

        return "\n".join(lines)


def _extract_hermes_profile_and_task(command: str) -> tuple[Optional[str], Optional[str]]:
    profile: Optional[str] = None
    try:
        tokens = shlex.split(command)
    except Exception:
        tokens = command.split()
    for index, token in enumerate(tokens):
        if token in {"-p", "--profile"} and index + 1 < len(tokens):
            profile = tokens[index + 1]
            break
        if token.startswith("--profile="):
            profile = token.split("=", 1)[1]
            break
    if profile and not any(_is_hermes_token(token) for token in tokens):
        profile = None
    return profile, _extract_kanban_task(command)


def _is_hermes_token(token: str) -> bool:
    return token == "hermes" or token.endswith("/hermes")


def _extract_kanban_task(text: str) -> Optional[str]:
    match = _KANBAN_TASK_RE.search(str(text or ""))
    return match.group(1) if match else None


def _summarize_command_shape(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except Exception:
        tokens = command.split()
    if not tokens:
        return "Commande en cours"
    # Skip VAR=value environment prefixes to show the executable, not secrets or env noise.
    command_token = next((tok for tok in tokens if "=" not in tok or tok.startswith(('/', './'))), tokens[0])
    executable = command_token.rsplit("/", 1)[-1]
    if executable in {"python", "python3"}:
        return "Script Python en shell"
    if executable in {"git", "gh"}:
        return "Opération Git/GitHub"
    if executable in {"yarn", "npm", "pnpm", "node"}:
        return "Opération Node/Yarn"
    if executable in {"pytest", "python -m pytest"}:
        return "Tests automatisés"
    return f"Commande {executable}"


def _clean_preview(value: Any) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = _SECRETISH_RE.sub("[secret]", text)
    return " ".join(text.split())


def _safe_text(value: Any) -> str:
    return _shorten(_clean_preview(value), 90)


def _shorten(value: Any, limit: int = 80) -> str:
    text = _clean_preview(value)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _format_duration(value: Any) -> str:
    try:
        seconds = float(value)
    except Exception:
        return ""
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes / 60:.1f} h"


__all__ = ["ToolProgressSummary"]
