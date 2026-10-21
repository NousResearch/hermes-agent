"""Size signals for committed skill writes in the self-improvement review footer."""

from pathlib import Path

from agent.i18n import t


def skill_size_warning_lines(data: dict) -> list[str]:
    if data.get("staged"):
        return []
    if "results" not in data:
        rows = [data]
    elif data.get("operations_applied") and isinstance(data["results"], list):
        rows = data["results"]
    else:
        return []
    warnings = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("success") is not True:
            continue
        warning = row.get("size_warning")
        if (isinstance(warning, dict) and isinstance(warning.get("name"), str)
                and isinstance(warning.get("chars"), int)):
            warnings[warning["name"]] = warning
        elif row.get("action") in {"create", "patch", "edit", "write_file"}:
            path = row.get("file_path")
            if not path or Path(path).parts == ("SKILL.md",):
                # A later main-file write below the threshold clears the intermediate signal.
                warnings.pop(row.get("name"), None)
    return [t("display.review.skill_size", name=w["name"], chars=w["chars"])
            for w in warnings.values()]
