import sys; sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")  # usage: python hardline_scanner_matrix.py <repo_root>
from tools.approval_detection import detect_hardline_command as d
cases = {
    # the reviewer's witnesses
    "newline-hidden reboot in quoted $(grep)":  ('echo "$(grep -P \'safe\' /dev/null\nreboot)"', True),
    "grep with backtick operand":                ("grep -e `echo needle` file", False),
    # the original class (must stay fixed)
    "canonical sed/grep/cut":                    ('sed -n "$(grep -n X f | cut -d: -f1),+3p" f', False),
    "grep -c inside $()":                        ('echo "$(grep -c x f)"', False),
    # controls: other separators inside the substitution
    "; hidden reboot":                           ('echo "$(grep x f; reboot)"', True),
    "&& hidden reboot":                          ('echo "$(grep x f && reboot)"', True),
    "| hidden shutdown":                         ('echo "$(grep x f | shutdown -h now)"', True),
    "nested $( ) newline reboot":                ('echo "$(echo $(grep x f)\nreboot)"', True),
    "backtick newline reboot":                   ('echo "`grep x f\nreboot`"', True),
    # data that must stay allowed
    "reboot as quoted grep pattern":             ("grep -F 'sudo reboot' notes.md", False),
    "commit msg with reboot on a data line":     ('git commit -m "fix\nsudo reboot handling"', False),
}
bad = 0
for name, (cmd, want_block) in cases.items():
    got, why = d(cmd)
    ok = got == want_block
    bad += not ok
    print(("OK  " if ok else "FAIL") + f" {name}: blocked={got} ({why})")
print("ALL OK" if not bad else f"{bad} FAILURES")
