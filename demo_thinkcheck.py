import sys
sys.path.insert(0, 'd:\\luoxuejian000\\new01\\hermes-agent-main\\hermes-agent-main')

from tools.thinkcheck_tool import evaluate_text

print('=' * 70)
print('ThinkCheck × Hermes Agent 集成演示')
print('=' * 70)

# 测试文本1（法律）
print()
print('=' * 70)
print('【测试文本1 - 法律领域】')
print('=' * 70)
text1 = '根据股权转让协议，我方已于去年完成全部出资义务，因此我方持有公司51%的股权。所以，我方并未持有公司多数股权。'
print(text1)
print()

result1 = evaluate_text(text1)
print('【评估结果】')
print(f"  U(统一性) = {result1['U']:.4f}")
print(f"  D(发展性) = {result1['D']:.4f}")
print(f"  A(对抗性) = {result1['A']:.4f}")
print(f"  H(和谐度) = {result1['H']:.4f}")
print(f"  通俗解读: {result1['verdict']}")

# 测试文本2（金融）
print()
print('=' * 70)
print('【测试文本2 - 金融领域】')
print('=' * 70)
text2 = '根据估值分析，该公司的市盈率为15倍，处于行业平均水平。公司现金流稳定，资产结构合理，负债率在可控范围内。因此，我们建议买入该公司股票，目标价上调至50元。但是，我们也需要注意市场整体风险和行业政策变化。'
print(text2)
print()

result2 = evaluate_text(text2)
print('【评估结果】')
print(f"  U(统一性) = {result2['U']:.4f}")
print(f"  D(发展性) = {result2['D']:.4f}")
print(f"  A(对抗性) = {result2['A']:.4f}")
print(f"  H(和谐度) = {result2['H']:.4f}")
print(f"  通俗解读: {result2['verdict']}")

print()
print('=' * 70)
print('演示完成！ThinkCheck 3.0 已成功集成到 Hermes Agent')
print('=' * 70)