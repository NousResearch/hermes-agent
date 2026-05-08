import sys
print('Python:', sys.version)
sys.path.insert(0, '.')
print('Path:', sys.path[0])

try:
    from thinkcheck_harmony import HarmonyEvaluator
    print('Import HarmonyEvaluator: OK')
except Exception as e:
    print('Import HarmonyEvaluator failed:', e)

try:
    from tools.thinkcheck_tool import evaluate_text
    print('Import evaluate_text: OK')
except Exception as e:
    print('Import evaluate_text failed:', e)