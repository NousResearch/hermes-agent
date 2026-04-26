"""Feishu CardKit JSON builder -- chainable API for Interactive Message Cards.

Ports the openclaw-lark card/builder.js logic to Python.
Produces JSON compatible with lark_oapi's MessageCreateRequest (message_v2).

Typical usage::

    card = (
        CardBuilder()
        .header("My Title", color="blue")
        .add_markdown("**Hello** world")
        .add_divider()
        .add_button("Click me", action={"type": "url", "url": "https://example.com"})
        .build()
    )
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Status / colour constants
# ---------------------------------------------------------------------------

COLORS = {
    "green": "green",
    "red": "red",
    "yellow": "yellow",
    "orange": "orange",
    "blue": "blue",
    "grey": "grey",
    "turquoise": "turquoise",
    "purple": "purple",
    "indigo": "indigo",
    "wathet": "wathet",
    "carmine": "carmine",
    "violet": "violet",
    "lime": "lime",
}

# Canonical header template colours for card status
CARD_COLORS = {
    "success": "green",
    "error": "red",
    "warning": "orange",
    "info": "blue",
    "default": "grey",
}

# Tool-use step status display map
_TOOL_STATUS_MAP: dict[str, dict[str, str]] = {
    "running": {"label": "Running", "color": "turquoise"},
    "error": {"label": "Failed", "color": "red"},
    "success": {"label": "Succeeded", "color": "green"},
}

_TOOL_USE_STEP_INDENT = "0px 0px 0px 22px"

# ---------------------------------------------------------------------------
# Small helpers (mirrors JS utilities)
# ---------------------------------------------------------------------------


def format_elapsed(ms: float) -> str:
    """Convert milliseconds to a human-readable duration string.

    Args:
        ms: Duration in milliseconds.

    Returns:
        E.g. ``"3.2s"`` or ``"2m 5s"``.
    """
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = round(seconds % 60)
    return f"{minutes}m {rem}s"


def compact_number(value: float) -> str:
    """Format a number compactly with k/m suffix.

    Args:
        value: Numeric value to format.

    Returns:
        E.g. ``"1.2k"``, ``"3m"``, ``"42"``.
    """
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        m = value / 1_000_000
        return f"{round(m)}m" if abs(m) >= 100 else f"{m:.1f}m"
    if abs_val >= 1_000:
        k = value / 1_000
        return f"{round(k)}k" if abs(k) >= 100 else f"{k:.1f}k"
    return str(round(value))


def _escape_md(value: str) -> str:
    """Escape special Markdown characters for Feishu lark_md.

    Args:
        value: Raw text to escape.

    Returns:
        Escaped text safe for use inside Markdown spans.
    """
    value = value.replace("\\", "\\\\")
    return re.sub(r"([`*_{}\[\]<>])", r"\\\1", value)


def _tool_status(status: str) -> dict[str, str]:
    return _TOOL_STATUS_MAP.get(status, _TOOL_STATUS_MAP["success"])


def _longest_backtick_run(text: str) -> int:
    runs = re.findall(r"`+", text)
    return max((len(r) for r in runs), default=0)


def _format_code_block(content: str, language: str = "") -> str:
    normalized = content.replace("\r\n", "\n").strip()
    fence = "`" * max(3, _longest_backtick_run(normalized) + 1)
    return f"{fence}{language}\n{normalized}\n{fence}"


# ---------------------------------------------------------------------------
# CardBuilder
# ---------------------------------------------------------------------------


class CardBuilder:
    """Chainable builder that constructs a Feishu Interactive Message Card dict.

    Follows the Feishu message_v2 / CardKit card schema:
    https://open.feishu.cn/document/uAjLw4CM/ukzMukzMukzM/feishu-cards/overview

    All mutating methods return ``self`` to allow fluent chaining.

    Example::

        card_dict = (
            CardBuilder()
            .header("Deployment complete", color="success")
            .add_markdown("**Status:** all green")
            .add_divider()
            .add_button("View logs", action={"type": "url", "url": "https://..."})
            .build()
        )
    """

    def __init__(self) -> None:
        self._header: Optional[dict[str, Any]] = None
        self._elements: list[dict[str, Any]] = []
        self._config: dict[str, Any] = {
            "wide_screen_mode": True,
            "update_multi": True,
            "locales": ["zh_cn", "en_us"],
        }

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def header(
        self,
        title: str,
        color: str = "blue",
        subtitle: Optional[str] = None,
    ) -> "CardBuilder":
        """Set the card header with a title and optional template colour.

        Args:
            title: Plain-text title shown in the card header.
            color: Template colour name.  Accepts a raw Feishu colour token
                (e.g. ``"green"``) or a semantic alias from :data:`CARD_COLORS`
                (``"success"``, ``"error"``, ``"warning"``, ``"info"``).
                Defaults to ``"blue"``.
            subtitle: Optional plain-text subtitle line below the title.

        Returns:
            ``self`` for chaining.
        """
        template = CARD_COLORS.get(color, color)
        title_elem: dict[str, Any] = {"tag": "plain_text", "content": title}
        self._header = {
            "title": title_elem,
            "template": template,
        }
        if subtitle:
            self._header["subtitle"] = {"tag": "plain_text", "content": subtitle}
        return self

    # ------------------------------------------------------------------
    # Block builders
    # ------------------------------------------------------------------

    def add_text(self, content: str) -> "CardBuilder":
        """Append a plain-text block (lark_md div).

        Args:
            content: Plain text string to display.

        Returns:
            ``self`` for chaining.
        """
        self._elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": content},
            }
        )
        return self

    def add_markdown(self, md: str) -> "CardBuilder":
        """Append a Markdown block.

        Args:
            md: Markdown-formatted string.  Feishu lark_md syntax is supported.

        Returns:
            ``self`` for chaining.
        """
        self._elements.append({"tag": "markdown", "content": md})
        return self

    def add_divider(self) -> "CardBuilder":
        """Append a horizontal rule divider.

        Returns:
            ``self`` for chaining.
        """
        self._elements.append({"tag": "hr"})
        return self

    def add_image(
        self,
        url: str,
        alt_text: str = "",
        img_key: Optional[str] = None,
    ) -> "CardBuilder":
        """Append an image block.

        Provide either ``url`` (external) or ``img_key`` (uploaded to Feishu).
        When both are given ``img_key`` takes precedence.

        Args:
            url: Public image URL.
            alt_text: Accessibility alt text shown when the image cannot load.
            img_key: Feishu image key (from file upload API).

        Returns:
            ``self`` for chaining.
        """
        elem: dict[str, Any] = {
            "tag": "img",
            "alt": {"tag": "plain_text", "content": alt_text},
        }
        if img_key:
            elem["img_key"] = img_key
        else:
            elem["img_key"] = url  # external URL stored in img_key per Feishu docs
        self._elements.append(elem)
        return self

    def add_button(
        self,
        text: str,
        action: Optional[dict[str, Any]] = None,
        button_type: str = "default",
    ) -> "CardBuilder":
        """Append a single action button.

        Wraps the button in an ``action`` container element.

        Args:
            text: Button label text.
            action: Feishu button value/action dict (e.g.
                ``{"type": "url", "url": "https://..."}`` or a callback value
                dict for card callbacks).
            button_type: Feishu button type token: ``"primary"``,
                ``"danger"``, or ``"default"``.

        Returns:
            ``self`` for chaining.
        """
        btn: dict[str, Any] = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": text},
            "type": button_type,
        }
        if action is not None:
            btn["value"] = action
        self._elements.append({"tag": "action", "actions": [btn]})
        return self

    def add_tool_use(
        self,
        tool_name: str,
        status: str = "success",
        summary: Optional[str] = None,
        detail: Optional[str] = None,
        elapsed_ms: Optional[float] = None,
    ) -> "CardBuilder":
        """Append a collapsible tool-call step panel.

        Renders a compact representation of a single tool invocation,
        mirroring the ``buildToolUsePanel`` logic in builder.js.

        Args:
            tool_name: Display name of the tool (e.g. ``"web_search"``).
            status: One of ``"running"``, ``"success"``, or ``"error"``.
            summary: One-line summary shown beneath the tool title.
            detail: Additional detail text (plain, grey, notation size).
            elapsed_ms: Execution duration in milliseconds; included in the
                panel title when provided.

        Returns:
            ``self`` for chaining.
        """
        st = _tool_status(status)
        title_md = (
            f"**{_escape_md(tool_name)}**"
            f" · <font color='{st['color']}'>{st['label']}</font>"
        )
        if elapsed_ms is not None and elapsed_ms > 0:
            d = format_elapsed(elapsed_ms)
            title_md += f" · {d}"

        step_elements: list[dict[str, Any]] = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": title_md,
                    "text_size": "notation",
                },
            }
        ]

        if detail:
            step_elements.append(
                {
                    "tag": "div",
                    "margin": _TOOL_USE_STEP_INDENT,
                    "text": {
                        "tag": "plain_text",
                        "content": detail.strip(),
                        "text_color": "grey",
                        "text_size": "notation",
                    },
                }
            )

        if summary:
            step_elements.append(
                {
                    "tag": "div",
                    "margin": _TOOL_USE_STEP_INDENT,
                    "text": {
                        "tag": "lark_md",
                        "content": summary.strip(),
                        "text_size": "notation",
                    },
                }
            )

        panel: dict[str, Any] = {
            "tag": "collapsible_panel",
            "expanded": status == "running",
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"🛠️ {tool_name}",
                    "text_color": "grey",
                    "text_size": "notation",
                },
                "vertical_align": "center",
                "icon": {
                    "tag": "standard_icon",
                    "token": "down-small-ccm_outlined",
                    "color": "grey",
                    "size": "16px 16px",
                },
                "icon_position": "right",
                "icon_expanded_angle": -180,
            },
            "border": {"color": "grey", "corner_radius": "5px"},
            "vertical_spacing": "4px",
            "padding": "8px 8px 8px 8px",
            "elements": step_elements,
        }
        self._elements.append(panel)
        return self

    def add_note(self, content: str, color: str = "grey") -> "CardBuilder":
        """Append a notation-sized footnote markdown block.

        Args:
            content: Markdown text to render at notation (small) size.
            color: Optional font colour wrapping the content.

        Returns:
            ``self`` for chaining.
        """
        if color and color != "default":
            body = f"<font color='{color}'>{content}</font>"
        else:
            body = content
        self._elements.append(
            {
                "tag": "markdown",
                "content": body,
                "text_size": "notation",
            }
        )
        return self

    def add_collapsible(
        self,
        title: str,
        elements: list[dict[str, Any]],
        expanded: bool = False,
    ) -> "CardBuilder":
        """Append a collapsible panel with pre-built inner elements.

        Args:
            title: Plain-text panel header title.
            elements: List of Feishu element dicts for the panel body.
            expanded: Whether the panel starts expanded.  Defaults to
                ``False``.

        Returns:
            ``self`` for chaining.
        """
        panel: dict[str, Any] = {
            "tag": "collapsible_panel",
            "expanded": expanded,
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "vertical_align": "center",
                "icon": {
                    "tag": "standard_icon",
                    "token": "down-small-ccm_outlined",
                    "size": "16px 16px",
                },
                "icon_position": "follow_text",
                "icon_expanded_angle": -180,
            },
            "border": {"color": "grey", "corner_radius": "5px"},
            "vertical_spacing": "8px",
            "padding": "8px 8px 8px 8px",
            "elements": elements,
        }
        self._elements.append(panel)
        return self

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def set_config(self, **kwargs: Any) -> "CardBuilder":
        """Override or extend top-level card config keys.

        Args:
            **kwargs: Key-value pairs merged into the card ``config`` dict.

        Returns:
            ``self`` for chaining.
        """
        self._config.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # build()
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Serialise the card to a Feishu Interactive Message Card dict.

        The returned dict can be serialised with ``json.dumps`` and passed
        to ``lark_oapi`` as the card body.

        Returns:
            A ``dict`` with keys ``config``, ``elements``, and optionally
            ``header``.
        """
        card: dict[str, Any] = {
            "config": self._config,
            "elements": list(self._elements),
        }
        if self._header is not None:
            card["header"] = self._header
        return card

    def build_v2(self) -> dict[str, Any]:
        """Serialise to CardKit 2.0 schema (``schema: "2.0"``).

        Uses ``body.elements`` instead of top-level ``elements``, and omits
        the ``wide_screen_mode`` / ``update_multi`` keys that are invalid
        in schema 2.0.

        Returns:
            A ``dict`` with keys ``schema``, ``config``, ``body``, and
            optionally ``header``.
        """
        config_v2 = {
            k: v
            for k, v in self._config.items()
            if k not in {"wide_screen_mode", "update_multi"}
        }
        card: dict[str, Any] = {
            "schema": "2.0",
            "config": config_v2,
            "body": {"elements": list(self._elements)},
        }
        if self._header is not None:
            card["header"] = self._header
        return card


# ---------------------------------------------------------------------------
# Module-level convenience factories
# ---------------------------------------------------------------------------


def thinking_card() -> dict[str, Any]:
    """Return a minimal 'Thinking...' card for the initial streaming state.

    Returns:
        Feishu card dict with a single markdown element.
    """
    return CardBuilder().add_markdown("Thinking...").build()


def error_card(message: str) -> dict[str, Any]:
    """Return a red-header error card with the given message.

    Args:
        message: Human-readable error description.

    Returns:
        Feishu card dict.
    """
    return (
        CardBuilder()
        .header("Error", color="error")
        .add_markdown(f"<font color='red'>{message}</font>")
        .build()
    )
