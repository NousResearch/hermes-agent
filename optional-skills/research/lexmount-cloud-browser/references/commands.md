# 命令与示例

每条命令返回 JSON：成功时包含 `ok: true` 和 `data`，失败时包含 `ok: false`、`error` 和错误信息。成功表示命令执行完成；具体网页动作仍须检查实际结果。

以下 `browser-cli` 均为技能目录中二进制的简写，调用时替换为主文档中对应平台的绝对入口。

## 命令速查

方括号表示可选参数，不要原样输入；`ID`、`URL`、`CSS`、`FILE`、`JS` 等是待替换值，分别表示标识、网址、选择器、文件路径和 JavaScript 表达式。带空格的文本或路径应加引号。

```text
browser-cli doctor
browser-cli auth status
browser-cli auth login [--client-name "NAME"]

browser-cli session create [--browser-mode normal|light]
  [--context-id ID --context-mode read_write|read_only]
  [--downloads] [--recording] [--window-size 1920,1080]
browser-cli session list [--status active]
browser-cli session get --session-id ID
browser-cli session targets --session-id ID
browser-cli session keepalive --session-id ID [--duration 60]
browser-cli session close --session-id ID
browser-cli session downloads list --session-id ID
browser-cli session downloads get --session-id ID --download-id ID --output FILE
browser-cli session downloads archive --session-id ID --output FILE
browser-cli session downloads delete --session-id ID --yes

browser-cli context create [--description TEXT] [--metadata-json JSON]
browser-cli context list [--status available|locked] [--limit 20]
browser-cli context get --context-id ID
browser-cli context fork --context-id ID
browser-cli context delete --context-id ID --yes
browser-cli context force-release --context-id ID --yes

browser-cli action open-url --session-id ID --url URL
browser-cli action snapshot --session-id ID
browser-cli action wait-selector --session-id ID --selector CSS
browser-cli action wait-text --session-id ID --text "Saved" [--selector CSS]
browser-cli action wait-text --session-id ID --text "Saved" [--selector CSS] --exact
browser-cli action click --session-id ID --selector CSS
browser-cli action fill --session-id ID --selector CSS --value TEXT
browser-cli action screenshot --session-id ID --path FILE [--full-page]
browser-cli action pdf --session-id ID --path FILE [--print-background]
browser-cli action eval --session-id ID --expression JS
browser-cli action raw --session-id ID --method CDP_METHOD --params-json JSON
```

公开浏览使用临时会话。需要复用登录时，每个账号或用途使用专用 Context；不要让并行任务共享同一个读写 Context。

`wait-text` 默认忽略大小写、归一化后做包含匹配。仅当整段文本必须相等时使用 `--exact`，不支持旧写法 `--match contains`。

## 浏览页面与读取信息

将创建结果中的 `data.session_id` 填入后续命令的 `ID`：

```text
browser-cli session create
browser-cli action open-url --session-id ID --url "https://example.com"
browser-cli action snapshot --session-id ID
```

从实际返回内容读取标题、正文和链接。需要继续浏览时，根据页面中的真实链接或元素导航，再读取页面状态；按用户问题整理信息并注明来源。

用户需要截图时，在关闭会话前执行以下命令，将 `FILE` 替换为当前任务内实际的 PNG 文件路径：

```text
browser-cli action screenshot --session-id ID --path FILE --full-page
```

截图后核验文件存在且非空，再交付文件。任务结束时执行 `browser-cli session close --session-id ID`；网页失败时也要清理本次临时会话，正在人工接管时除外。

## 填写表单与读回

以 `https://www.selenium.dev/selenium/web/web-form.html` 为例，先打开页面并读取 snapshot，再确认下面的选择器仍然适用。用户只要求填写时，不点击 Submit。

```text
browser-cli action fill --session-id ID --selector "#my-text-id" --value "Cloud Browser Test"
browser-cli action fill --session-id ID --selector "textarea[name=my-textarea]" --value "This is a browser test"
```

该页下拉框显示文字 `Two` 对应的 option value 是 `2`。此版 `fill` 直接写入元素 value，不按显示文字查找选项，且不会验证最终选中状态。应先从页面确认实际选项值，再赋值并读回，不直接把标签 `Two` 当成 value。

需要用 `action eval --expression JS` 操作下拉框时，下面是该示例页的完整 JS 表达式；整体作为一个参数传递，并按当前 shell 正确引用：

```javascript
(() => {
  const select = document.querySelector('select[name=my-select]');
  if (!select) throw new Error('select not found');
  const option = Array.from(select.options).find(item => item.value === '2');
  if (!option) throw new Error('option not found');
  select.value = option.value;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  select.dispatchEvent(new Event('change', { bubbles: true }));
  if (select.value !== option.value) throw new Error('selection did not take effect');
  return { value: select.value, label: select.options[select.selectedIndex]?.textContent ?? null };
})()
```

随后可分别读回字段值；示例输出应为 `Cloud Browser Test`、`This is a browser test` 和 `2`：

```text
browser-cli action eval --session-id ID --expression "document.querySelector('#my-text-id').value"
browser-cli action eval --session-id ID --expression "document.querySelector('textarea[name=my-textarea]').value"
browser-cli action eval --session-id ID --expression "document.querySelector('select[name=my-select]').value"
```

如果读回不符合预期，说明尚未完成；不要把命令成功当成任务成功。核验后交付填写结果；如用户需要，再截图。完成后关闭临时会话。

## eval 的用法边界

`eval` 接收 JavaScript 表达式，不自动包装函数。`document.title` 可直接使用；需要多步并返回对象时，使用上例的立即调用函数 `(() => { ...; return result; })()`，不要写顶层 `return`。

读取元素前检查是否存在；读取选项文字前处理 `selectedIndex === -1` 的情况。`Uncaught` 可能来自脚本语法或属性访问错误，不代表云浏览器必然断线。
