import os

def find_cns_core():
    # Search all local drives and common backup locations for cosmosynapse engine
    search_dirs = [
        r"d:\Cosmos",
        r"d:\Cosmos",
        r"C:\Users\corys\.gemini"
    ]
    
    for base_dir in search_dirs:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py') or file.endswith('.txt') or file.endswith('.md'):
                    try:
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if 'class CosmosCNS' in content and 'def process_user_input' in content:
                                print(f"FOUND: {filepath}")
                                # Print first 50 lines to verify it's the right one
                                print("--- PREVIEW ---")
                                lines = content.split('\n')
                                for i in range(min(50, len(lines))):
                                    print(lines[i])
                                print("---------------")
                                return
                    except Exception:
                        pass
    print("NOT FOUND")

if __name__ == "__main__":
    find_cns_core()
