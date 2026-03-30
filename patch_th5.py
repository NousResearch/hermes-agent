import re
with open("agent/google_adapter.py", "r") as f:
    lines = f.readlines()
    
for i, line in enumerate(lines):
    if "extra = tc.get(" in line:
        lines[i] = "                extra = tc.get('extra_content', {})\n"
        
with open("agent/google_adapter.py", "w") as f:
    f.writelines(lines)
print("done")
