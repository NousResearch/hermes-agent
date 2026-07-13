"""Agent i18n — localized system messages.

Provides translated strings for system-generated messages injected
into system prompts and tool results.  Extensible: add a dict per
locale (id, ja, ko, fr, de, ...).

Config:  config.yaml → agent.lang: "id"
         or env: HERMES_LANG=id
"""

from __future__ import annotations

import os
from typing import Dict

# ── Lookup ──────────────────────────────────────────────────────────────

_CURRENT_LANG = os.environ.get("HERMES_LANG", "")

# ── Translations ────────────────────────────────────────────────────────

_MESSAGES: Dict[str, Dict[str, str]] = {
    "en": {
        # Task resume
        "task_resume_header": "[TASK RESUME]",
        "task_resume_body": "You are resuming a previously interrupted task.",
        "task_resume_original": "Original task",
        "task_resume_phase": "Current phase",
        "task_resume_progress": "Progress: {count} steps completed (do NOT re-execute):",
        "task_resume_budget": "Budget: {count} turns remaining.",
        "task_resume_continue": "Continue from where you left off. Verify state before repeating any action.",
        # Mailbox
        "mailbox_header": "[AGENT MAILBOX — {count} unread message(s)]",
        "mailbox_age_s": "{age}s ago",
        "mailbox_age_m": "{age}m ago",
        # Error recovery
        "skill_patch_hint": (
            "After you recover from this failure, consider patching "
            "the relevant skill with the learned fix using "
            "skill_manage(action='patch'). Add the pitfall to the "
            "skill's ## Pitfalls section so future sessions avoid "
            "this issue."
        ),
        # Dashboard
        "dashboard_health": "Health",
        "dashboard_tools": "Tools",
        "dashboard_budget": "Budget",
        "dashboard_phase": "Phase",
        "dashboard_done": "Done",
        "dashboard_mailbox": "Mailbox",
        "dashboard_errors": "Error breakdown",
        "dashboard_events": "Recent events",
        "dashboard_no_task": "(no active task)",
        "dashboard_no_errors": "No errors recorded",
        "dashboard_no_events": "(no events yet)",
    },
    "id": {
        # Task resume
        "task_resume_header": "[LANJUTKAN TUGAS]",
        "task_resume_body": "Anda melanjutkan tugas yang sebelumnya terhenti.",
        "task_resume_original": "Tugas awal",
        "task_resume_phase": "Fase saat ini",
        "task_resume_progress": "Progress: {count} langkah selesai (JANGAN ulangi):",
        "task_resume_budget": "Budget: {count} giliran tersisa.",
        "task_resume_continue": "Lanjutkan dari tempat Anda berhenti. Verifikasi kondisi sebelum mengulangi tindakan apa pun.",
        # Mailbox
        "mailbox_header": "[KOTAK MASUK AGEN — {count} pesan belum dibaca]",
        "mailbox_age_s": "{age} detik lalu",
        "mailbox_age_m": "{age} menit lalu",
        # Error recovery
        "skill_patch_hint": (
            "Setelah Anda pulih dari kegagalan ini, pertimbangkan "
            "untuk memperbarui skill terkait dengan perbaikan yang "
            "dipelajari menggunakan skill_manage(action='patch'). "
            "Tambahkan ke bagian ## Pitfalls skill agar sesi "
            "selanjutnya tidak mengalami masalah ini."
        ),
        # Dashboard
        "dashboard_health": "Kesehatan",
        "dashboard_tools": "Tools",
        "dashboard_budget": "Budget",
        "dashboard_phase": "Fase",
        "dashboard_done": "Selesai",
        "dashboard_mailbox": "Kotak Masuk",
        "dashboard_errors": "Rincian Error",
        "dashboard_events": "Kejadian Terbaru",
        "dashboard_no_task": "(tidak ada tugas aktif)",
        "dashboard_no_errors": "Tidak ada error tercatat",
        "dashboard_no_events": "(belum ada kejadian)",
    },
}


# ── Public API ──────────────────────────────────────────────────────────


def set_lang(lang: str) -> None:
    """Set the active language for system messages."""
    global _CURRENT_LANG
    _CURRENT_LANG = lang


def get_lang() -> str:
    """Return current language code, defaulting to 'en'."""
    return _CURRENT_LANG or "en"


def t(key: str, **kwargs) -> str:
    """Translate a message key to the active language.

    Falls back to English if the key or language is missing.
    """
    lang = get_lang()
    if lang not in _MESSAGES:
        lang = "en"
    msg = _MESSAGES.get(lang, {}).get(key)
    if msg is None:
        msg = _MESSAGES.get("en", {}).get(key, key)
    if kwargs:
        try:
            msg = msg.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return msg


def lang_directive(lang: str) -> str:
    """Return a system-prompt language directive for the given locale."""
    directives = {
        "id": (
            "You are speaking with an Indonesian user (Bahasa Indonesia). "
            "Respond in Indonesian unless the user writes in English. "
            "Translate all system guidance and tool descriptions mentally "
            "to Indonesian before acting. Keep technical terms in English "
            "where no natural Indonesian equivalent exists."
        ),
        "ja": (
            "You are speaking with a Japanese user. "
            "Respond in Japanese unless the user writes in English. "
            "Translate all system guidance and tool descriptions mentally "
            "to Japanese before acting."
        ),
        "ko": (
            "You are speaking with a Korean user. "
            "Respond in Korean unless the user writes in English."
        ),
    }
    return directives.get(lang, directives.get("en", ""))