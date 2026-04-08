import sys

file_path = r'C:\Users\beer8\AppData\Local\hermes\hermes-agent\hermes_cli\doctor.py'

replacements = {
    '◆': '*',
    '✓': '[OK]',
    '✗': '[FAIL]',
    '⚠': '[WARN]',
    '\u26a0': '[WARN]', # Literal warning sign
    '\u2713': '[OK]',   # Literal checkmark
    '\u2717': '[FAIL]', # Literal x-mark
}

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully replaced all UTF-8 symbols in doctor.py")
except Exception as e:
    print(f"Error: {e}")
