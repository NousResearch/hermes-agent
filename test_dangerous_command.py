from tools.approval import detect_dangerous_command

command = "rm -rf /important"
is_dangerous, pattern_key, description = detect_dangerous_command(command)
print(f"Command: {command}")
print(f"Is dangerous: {is_dangerous}")
print(f"Pattern key: {pattern_key}")
print(f"Description: {description}")
