"""The words of first-run onboarding: the scripted greeting, the opening prompt, and the reminder
that rides on every ``setup_beat`` result. Pure; ``reminder(key, record, facts)`` renders the key
``agent/onboarding_flow.py::apply`` returns.

The rules for the whole run live in the opening prompt (a hidden user turn, so the system prompt
stays byte-stable); reminders say only what was picked and what comes next.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent.onboarding_first_tasks import (
    FIXED_ID, MAX_LABEL, MAX_PICKS, OWN_TASK, TASK_BY_ID, eligible_tasks, task_skills,
)
from agent.onboarding_flow import Facts, next_beat
from tools.set_theme_tool import ACCENT_SWATCHES

CHAT_TITLE = "Getting started"
GREETING = "Hi, I'm Unitalk. Let's set things up for you. Then we'll get something cool done."

# The scripted opening in each desktop UI language (apps/desktop/src/i18n/languages.ts), played before
# the model speaks; the model writes the rest of the run in the user's language itself. Each greeting
# is the desktop's first-run intro copy (firstRun.intro in apps/desktop/src/i18n/*_first_run.ts), its
# two lines joined: keep the two equal.
_GREETINGS = {
    "en": GREETING,
    "ar": "مرحبًا، أنا Unitalk. لنجهّز كل شيء لك، ثم ننجز معًا شيئًا رائعًا.",
    "de": "Hallo, ich bin Unitalk. Richten wir alles für Sie ein. Danach machen wir etwas Cooles.",
    "es": "Hola, soy Unitalk. Vamos a dejarlo todo listo para ti. Luego haremos algo genial.",
    "fr": "Bonjour, je suis Unitalk. Configurons tout pour vous. Ensuite, nous ferons quelque chose de génial.",
    "ja": "こんにちは、Unitalk です。まずはあなたに合わせて設定しましょう。そのあと、何かすごいものを作りましょう。",
    "ru": "Привет, я Unitalk. Давайте всё настроим под вас. А потом сделаем что-нибудь классное.",
    "zh": "你好，我是 Unitalk。我们先帮你把一切设置好，然后一起做点很酷的事。",
    "zh-hant": "你好，我是 Unitalk。我們先幫你把一切設置好，然後一起做點很酷的事。",
}