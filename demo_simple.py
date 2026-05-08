import sys
sys.path.insert(0, '.')
from tools.thinkcheck_tool import evaluate_text

text1 = '根据股权转让协议，我方已于去年完成全部出资义务，因此我方持有公司51%的股权。所以，我方并未持有公司多数股权。'
result1 = evaluate_text(text1)
print('文本1评估结果:')
print('U=%.4f D=%.4f A=%.4f H=%.4f' % (result1['U'], result1['D'], result1['A'], result1['H']))
print('解读:', result1['verdict'])

text2 = '根据估值分析，该公司的市盈率为15倍，处于行业平均水平。公司现金流稳定，资产结构合理，负债率在可控范围内。因此，我们建议买入该公司股票，目标价上调至50元。但是，我们也需要注意市场整体风险和行业政策变化。'
result2 = evaluate_text(text2)
print('')
print('文本2评估结果:')
print('U=%.4f D=%.4f A=%.4f H=%.4f' % (result2['U'], result2['D'], result2['A'], result2['H']))
print('解读:', result2['verdict'])