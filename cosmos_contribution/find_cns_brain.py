import os

def find_cns_core():
    # Search the agent's brain for the previous implementation of CosmosCNS
    brain_dir = r"C:\Users\corys\.gemini\antigravity\brain"
    
    for root, dirs, files in os.walk(brain_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.md'):
                try:
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if 'class CosmosCNS' in line:
                                print(f"--- FOUND IN: {filepath} ---")
                                # Print surrounding lines to capture the class definition
                                start = max(0, i - 5)
                                end = min(len(lines), i + 200)
                                for j in range(start, end):
                                    print(lines[j].rstrip())
                                print("---------------------------")
                                return
                except Exception:
                    pass

if __name__ == "__main__":
    find_cns_core()
