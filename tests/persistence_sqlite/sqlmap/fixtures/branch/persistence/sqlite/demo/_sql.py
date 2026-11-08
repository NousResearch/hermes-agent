"""Backend fixture: shared SQL text."""

ITEM_COLUMNS = "id, name, size"


def _placeholders(count):
    return ", ".join("?" * count)
