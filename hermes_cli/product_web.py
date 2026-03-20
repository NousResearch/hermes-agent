from __future__ import annotations

import json

from hermes_cli.product_web_script import PAGE_SCRIPT
from hermes_cli.product_web_style import PAGE_STYLE
from hermes_cli.product_web_template import PAGE_TEMPLATE


DARK_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"></path></svg>'
LIGHT_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2.5M12 19.5V22M4.93 4.93l1.77 1.77M17.3 17.3l1.77 1.77M2 12h2.5M19.5 12H22M4.93 19.07l1.77-1.77M17.3 6.7l1.77-1.77"></path></svg>'


def build_product_index_html(*, product_name: str) -> str:
    safe_name = product_name.strip() or "Hermes Core"
    script = (
        PAGE_SCRIPT.replace("__PRODUCT_UI_CONFIG__", json.dumps({"productName": safe_name}))
        .replace("__DARK_ICON__", DARK_ICON)
        .replace("__LIGHT_ICON__", LIGHT_ICON)
    )
    return (
        PAGE_TEMPLATE.replace("__PRODUCT_NAME__", safe_name)
        .replace("__PAGE_STYLE__", PAGE_STYLE)
        .replace("__PAGE_SCRIPT__", script)
        .replace("__DARK_ICON__", DARK_ICON)
    )
