"""交互卡片基础组件。"""


def button(label, kind, value):
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": kind,
        "value": value,
    }


def card(title, template, markdown, *, actions=None):
    elements = [{"tag": "markdown", "content": markdown}]
    if actions is not None:
        elements.append({"tag": "action", "actions": actions})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"content": title, "tag": "plain_text"}, "template": template},
        "elements": elements,
    }
