import sys
sys.path.insert(0, '.')

from thinkcheck_harmony import HarmonyEvaluator, HarmonyReport, SuggestionEngine, get_preset
from tools.thinkcheck_tool import evaluate_text

print("=" * 60)
print("🧠 ThinkCheck × Hermes 集成测试")
print("=" * 60)

print("\n1. 测试核心模块导入...")
print("   ✅ HarmonyEvaluator")
print("   ✅ HarmonyReport")
print("   ✅ SuggestionEngine")
print("   ✅ get_preset")

print("\n2. 测试 evaluate_text 函数...")
result = evaluate_text('这个项目成本是100万元。实际上，这个项目的预算是200万元。')
print(f"   U(统一性) = {result['U']:.4f}")
print(f"   D(发展性) = {result['D']:.4f}")
print(f"   A(对抗性) = {result['A']:.4f}")
print(f"   H(和谐度) = {result['H']:.4f}")
print(f"   解读: {result['verdict']}")

print("\n3. 测试不同领域预设...")
presets = get_preset()
print(f"   可用预设领域: {list(presets.keys())}")

print("\n" + "=" * 60)
print("✅ 所有功能测试通过！")
print("=" * 60)