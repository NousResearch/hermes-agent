---
name: reliable-python-execution
description: Use this pattern when execute_code heredocs fail due to syntax errors or delimiter issues.
---

# Reliable Python Execution

## Trigger
When `execute_code` with `terminal(f'python3 << "EOF"\\n{code}\\nEOF')` results in `SyntaxError: unexpected character after line continuation character` or heredoc delimiter warnings.

## Steps
1. Write the Python code to a temporary file using `write_file(path='/tmp/script_name.py', content=code)`.
2. Execute the script via the `terminal` tool: `terminal(command='python3 /tmp/script_name.py')`.
3. (Optional) Clean up the temporary file after execution using `terminal(command='rm /tmp/script_name.py')`.

## Pitfalls
- **Heredoc Escaping:** Using formatted strings inside heredocs in `execute_code` often leads to shell interpretation errors or Python syntax errors if the string contains quotes or backslashes.
- **Persistence:** Files in `/tmp/` are generally safe for the duration of a session but should be uniquely named if parallel tasks are running.

## Verification
- Ensure the `terminal` output returns the expected stdout from the script without bash warning/error prefixes.
