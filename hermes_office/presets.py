"""Built-in role pictograms used in HireWizard step 2.

Each preset bundles: pictogram, default name pattern, default avatar, default
toolsets, default skills, default system prompt seed.  These are intentionally
*opinionated* and *safe* — they pick light toolsets so a kid clicking through
will get a working employee that doesn't, e.g., delete files.

Schema mirrored in ``frontend/src/types.ts``.
"""

from __future__ import annotations

from typing import TypedDict


class Preset(TypedDict):
    id: str
    label: str
    label_zh: str
    pictogram: str           # emoji shown on the big button
    default_name: str
    avatar_sprite: str
    default_hue: int
    toolsets: list[str]
    skills: list[str]
    system_prompt: str
    summary: str             # one-liner shown beneath the button


PRESETS: dict[str, Preset] = {
    "researcher": {
        "id": "researcher",
        "label": "Researcher",
        "label_zh": "研究员",
        "pictogram": "🔍",
        "default_name": "Researcher",
        "avatar_sprite": "scientist",
        "default_hue": 220,
        "toolsets": ["web", "file", "todo", "memory"],
        "skills": ["research/arxiv", "research/research-paper-writing"],
        "system_prompt": (
            "You are a meticulous research analyst. Investigate the user's "
            "question by searching the web, reading sources critically, "
            "and producing a sourced summary."
        ),
        "summary": "Reads the web and writes summaries with sources.",
    },
    "coder": {
        "id": "coder",
        "label": "Coder",
        "label_zh": "程序员",
        "pictogram": "👨‍💻",
        "default_name": "Coder",
        "avatar_sprite": "robot-1",
        "default_hue": 140,
        "toolsets": ["file", "code_execution", "todo"],
        "skills": [
            "software-development/test-driven-development",
            "software-development/systematic-debugging",
        ],
        "system_prompt": (
            "You are a careful software engineer. Read the requirement, "
            "write minimal code, run tests, and iterate."
        ),
        "summary": "Writes and tests small programs.",
    },
    "writer": {
        "id": "writer",
        "label": "Writer",
        "label_zh": "写作者",
        "pictogram": "✍️",
        "default_name": "Writer",
        "avatar_sprite": "writer",
        "default_hue": 320,
        "toolsets": ["file", "memory", "todo"],
        "skills": ["research/research-paper-writing"],
        "system_prompt": (
            "You are a clear, friendly writer. Match the requested tone, "
            "keep paragraphs short, and prefer concrete examples."
        ),
        "summary": "Drafts blogs, tweets, and emails in a clear voice.",
    },
    "designer": {
        "id": "designer",
        "label": "Designer",
        "label_zh": "设计师",
        "pictogram": "🎨",
        "default_name": "Designer",
        "avatar_sprite": "designer",
        "default_hue": 30,
        "toolsets": ["image_gen", "vision", "file"],
        "skills": ["creative/pixel-art", "creative/popular-web-designs"],
        "system_prompt": (
            "You are a visual designer. Translate ideas into images, "
            "explain choices, and iterate on feedback."
        ),
        "summary": "Generates images and visual concepts.",
    },
    "analyst": {
        "id": "analyst",
        "label": "Analyst",
        "label_zh": "分析师",
        "pictogram": "📊",
        "default_name": "Analyst",
        "avatar_sprite": "analyst",
        "default_hue": 200,
        "toolsets": ["code_execution", "file", "todo"],
        "skills": ["data-science/jupyter-live-kernel"],
        "system_prompt": (
            "You are a data analyst. Show your math, label your charts, "
            "and call out caveats."
        ),
        "summary": "Crunches numbers and explains the result.",
    },
    "translator": {
        "id": "translator",
        "label": "Translator",
        "label_zh": "翻译",
        "pictogram": "🌐",
        "default_name": "Translator",
        "avatar_sprite": "translator",
        "default_hue": 260,
        "toolsets": ["memory"],
        "skills": [],
        "system_prompt": (
            "You are a careful translator. Preserve tone and meaning, "
            "and flag culturally sensitive renderings."
        ),
        "summary": "Translates between languages.",
    },
    "tutor": {
        "id": "tutor",
        "label": "Tutor",
        "label_zh": "辅导老师",
        "pictogram": "📚",
        "default_name": "Tutor",
        "avatar_sprite": "tutor",
        "default_hue": 0,
        "toolsets": ["memory", "todo"],
        "skills": [],
        "system_prompt": (
            "You are a patient tutor. Explain concepts step-by-step, ask "
            "questions to check understanding, and never just dump answers."
        ),
        "summary": "Explains tough topics step by step.",
    },
    "helper": {
        "id": "helper",
        "label": "Helper",
        "label_zh": "万能助手",
        "pictogram": "🤝",
        "default_name": "Helper",
        "avatar_sprite": "robot-2",
        "default_hue": 60,
        "toolsets": ["web", "file", "todo", "memory"],
        "skills": [],
        "system_prompt": (
            "You are a friendly all-purpose helper. Take initiative; ask "
            "for clarification only when truly necessary."
        ),
        "summary": "A friendly all-purpose assistant.",
    },
}


def list_presets() -> list[Preset]:
    return list(PRESETS.values())


def get_preset(role_id: str) -> Preset | None:
    return PRESETS.get(role_id)
