import ast
from pathlib import Path


def validate_skill(skill_path: str) -> list[str]:
    """
    Scan all .py files inside a skill folder and check each one
    for Python syntax errors using ast.parse().

    Args:
        skill_path: Path to the skill directory (str or Path-like).

    Returns:
        A list of result strings, one per .py file found.
    """
    folder = Path(skill_path)
    py_files = list(folder.rglob("*.py"))

    if not py_files:
        return ["No Python files found."]

    results = []
    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            ast.parse(source, filename=str(py_file))
            results.append(f"OK: {py_file.name}")
        except SyntaxError as e:
            results.append(f"ERROR: {py_file.name} — {e}")
        except Exception as e:
            results.append(f"ERROR: {py_file.name} — unexpected error: {e}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python logic.py <path-to-skill-folder>")
        sys.exit(1)

    target = sys.argv[1]
    print(f"Validating Python files in: {target}\n")
    for line in validate_skill(target):
        print(line)
