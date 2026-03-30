with open("pyproject.toml", "r") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.strip() == '"pydantic>=2.12.5,<3",':
        lines.insert(i + 1, '  "google-genai>=0.2.2,<1",\n')
        break
with open("pyproject.toml", "w") as f:
    f.writelines(lines)
