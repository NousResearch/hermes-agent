# Bonus Activities Detection & Completion

Reference for finding and completing bonus activities on the Microsoft Rewards
earn page beyond the daily search/activity quota. Works with both Chinese (中文)
and English (英文) Bing Rewards.

## Detection

Navigate to `https://rewards.bing.com/earn` and run this JS to find all
actionable links:

```javascript
// Broad scan — all links with text
JSON.stringify(Array.from(document.querySelectorAll("a"))
  .map((b,i)=>({i,text:(b.innerText||"").replace(/\s+/g," ").trim().substring(0,120),href:b.href||""}))
  .filter(t=>t.text.length>5)
)
```

**Patterns to look for** (check text field — both Chinese and English):
- `+5`, `+10`, `+15`, `+50` — point values (locale-independent)
- `你是否知道答案`, `Bing quiz`, `Bing homepage quiz` — quiz link
- `完成此拼图`, `Complete the puzzle`, `imagepuzzle` — puzzle link
- `趋势`, `punchcard`, `0/N 个任务`, `0/N tasks` — punchcard campaigns
- `3 次搜索可得 10 分`, `3 searches = 10 points` — search bonus offer
- `冥想`, `海星`, `meditation`, `starfish` — topical search bonuses
- `立即激活`, `Activate now` — activation bonus links

## Completion by Type

### Search bonuses (+10 to +15)
Navigate directly to the href of the activity link via `page navigate`:

```python
subprocess.run([CUA_DRIVER, 'call', 'page'],
    input=json.dumps({'action':'navigate','pid':pid,'window_id':wid,
        'url': link_href}))
time.sleep(2)
```

### Quiz (+5)
Navigate to the quiz URL. Click answer option A (first answer):

```javascript
document.querySelectorAll("a").forEach(a=>{
    if(a.innerText.trim().match(/^A\./)) a.click();
});
```

The quiz has 3 questions. After answering one, click "Next" / "下一个":

```javascript
document.querySelectorAll("a,button,div,span").forEach(el=>{
    const t = (el.innerText||"").trim();
    if(t === "下一个" || t === "Next" || t === "Next event") el.click();
});
```

### Puzzle (+5)
Navigate to the puzzle URL — often auto-completes and shows "恭喜!" /
"Congratulations!" on load.

### Punchcard (+50 total)
Navigate to the quest/punchcard URL. Task labels differ by locale:

| Task | Chinese (中文) | English |
|------|---------------|---------|
| View schedule | 查看赛程 | View Schedule |
| View laptops | 查看笔记本电脑 | View Laptops |
| Discover deals | 发现优惠 | Discover Deals |

```javascript
Array.from(document.querySelectorAll("span")).filter(
    s => ["查看赛程","查看笔记本电脑","发现优惠",
          "View Schedule","View Laptops","Discover Deals"].includes(s.innerText.trim())
).forEach(s => s.click());
```

**Important**: Tasks 2-4 require 24h waits. Check for "等待 24 小时" or
"wait 24 hours" to confirm lock.

## Common patterns

Click punchcard task by button text:
```javascript
var spans = document.querySelectorAll("span");
for(var i=0;i<spans.length;i++){
    var t = spans[i].innerText.trim();
    if(["查看赛程","View Schedule","View Laptops","查看笔记本电脑"].includes(t)){
        spans[i].click();
        break;
    }
}
```

Read page text after an action:
```javascript
document.body.innerText.substring(0, 800).replace(/\s+/g, " ")
```
