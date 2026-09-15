# A4 页面平衡与视觉密度 QA

在浏览器完成基础打印预览后，用本规则判断简历是否只是“没有溢出”，还是已经形成可投递的页面节奏。页面密度指单位面积内可核验信息的密度，不是色块、卡片或边框的数量。先切换到 A4 分页模式，再测量 `.sheet`。

## 页面利用原则

- 内容能在正常字号下放进一页时，优先单页；不要因为模板预设了第二张 `.sheet` 就强制双页。
- 单页简历应形成完整的信息闭环，正文占用可用高度约 82%—96%；低于该范围通常需要重排，高于该范围要检查页尾拥挤。
- 多页简历的每个非末页应占用约 88%—98%，再进入下一页；禁止第一页明显空着、第二页却只有少量补充信息。
- 末页低于约 70% 时，先尝试把内容前移、合并低优先级分区或改回单页；只有长论文列表、作品集目录等结构性原因明确时才保留。
- 数值是诊断信号，不是机械评分。若内容完整、版面呼吸感合理，可以在交付说明中解释例外。

## 浏览器测量

在真实浏览器的分页模式执行：

```js
const pages = [...document.querySelectorAll('.sheet')].map((sheet, index) => {
  const root = sheet.querySelector(':scope > [contenteditable="true"]') || sheet;
  const rootStyle = getComputedStyle(root);
  const rootRect = root.getBoundingClientRect();
  const contentTop = rootRect.top + Number.parseFloat(rootStyle.paddingTop);
  const contentBottom = rootRect.bottom - Number.parseFloat(rootStyle.paddingBottom);
  const available = contentBottom - contentTop;
  const visibleContent = [...root.querySelectorAll('*')].filter((element) => {
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    const hasDirectText = [...element.childNodes]
      .some((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim());
    const isMedia = ['IMG', 'SVG', 'CANVAS', 'TABLE'].includes(element.tagName);
    return style.display !== 'none' && style.visibility !== 'hidden'
      && style.position !== 'fixed' && rect.width > 0 && rect.height > 0
      && (hasDirectText || isMedia);
  });
  const lastBottom = Math.max(contentTop, ...visibleContent.map((element) => element.getBoundingClientRect().bottom));
  const used = Math.max(0, lastBottom - contentTop);
  return {
    page: index + 1,
    fill: Math.round((used / available) * 100),
    overflowPx: Math.max(0, sheet.scrollHeight - sheet.clientHeight)
  };
});
console.table(pages);
```

同时保存整页截图并检查：视觉上是否头重脚轻、分区是否被拆散、最后一行是否贴近页底。DOM 数值和截图结论不一致时，以打印 PDF 的逐页渲染为准。

## 调整顺序

检测到过空或跨页失衡时，按以下顺序修改，前一步能解决就不要继续压缩：

1. **重排内容**：把最强项目和岗位关键词前置；把同类技能、奖项、链接合并为紧凑行；删除重复背景句和无法产生追问价值的弱要点。
2. **改变结构**：移除过早的手动分页；将低密度侧栏改为横向信息带；把次要经历压成一行贡献摘要，而不是平均分配篇幅。
3. **调整间距**：统一缩小过大的 section margin、公司条 padding 和项目间距，保留清晰的分区节奏。
4. **调整字号**：正文优先保持 9.5—10.5 pt、行距 1.24—1.36；不要为了单页低于 9 pt，也不要靠放大字号填空。
5. **决定扩页**：只有继续压缩会破坏可读性或证据完整性时才扩为两页，并重新平衡每页内容。

不得通过编造指标、重复技能、增加空泛自我评价或放大装饰元素来提高页面占用。

## 专业层级与装饰预算

- 技术、研究和通用岗位默认使用白底；主要依靠字号、字重、对齐、段前距和细分割线建立层级。
- 默认不使用满宽深色头部、卡片矩阵、成组徽章或为每个项目添加外框。除非用户明确要求作品集式视觉风格，否则这些元素会降低正文密度和专业感。
- 只使用一个主强调色，作用限于分区标题、链接、关键词和细线；正文保持深色，次要信息使用中性灰。
- 首屏 1/4 页内出现姓名、目标身份和一个最强证据即可；用文本顺序和粗细突出，不把三项信息分别装进色块。
- 项目使用稳定的“项目名 / 身份 / 日期—个人动作—验证结果”排版，但稳定不等于加框。优先使用同一基线、悬挂缩进和紧凑项目符号。
- 如果改版后相同内容占用的高度更多，或阅读顺序被装饰切碎，判定为视觉回退并撤销装饰。
- 保持文本可选择、可复制，避免复杂图表、进度条式技能评分、大面积背景和纯装饰图标。
- 把链接目的地视为信息而不是装饰：纸面版本应直接显示可读的规范化 URL、DOI 或域名/关键路径，不能只靠“网站”“PAPER”“CODE”等可点击标签。长地址可换行，但不能挤压、遮挡正文。

## 交付记录

交付说明写明：最终页数、各页占用比例、是否存在例外、PDF 纸张尺寸、纸面链接可识别性，以及逐页渲染检查结果。不要只写“已检查”。
