
import sys

def compare_files(f1, f2):
    with open(f1, 'r', encoding='utf-8') as file1, open(f2, 'r', encoding='utf-8') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
    
    maxlen = max(len(lines1), len(lines2))
    for i in range(maxlen):
        l1 = lines1[i].strip() if i < len(lines1) else "EOF"
        l2 = lines2[i].strip() if i < len(lines2) else "EOF"
        if l1 != l2:
            print(f"Difference at line {i+1}:")
            print(f"File 1: {l1}")
            print(f"File 2: {l2}")
            print("-" * 20)
    
    if len(lines1) != len(lines2):
        print(f"File lengths differ: L1={len(lines1)}, L2={len(lines2)}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare_files(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python compare_prompts.py file1 file2")
