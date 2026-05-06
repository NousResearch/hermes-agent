---
name: quotes-watchlist-manager
description: 维护 Hermes Web UI 的 /hermes/quotes 自选股（增删/替换/查看）并落盘保存；可同步刷新设置。
version: 0.1.0
platforms: [linux]
required_environment_variables:
  - name: HERMES_WEB_UI_BASE
    prompt: Hermes Web UI 基础地址（例如 http://127.0.0.1:8648）
    help: 该地址用于调用 hermes-web-ui 的自选股/偏好 API（/api/hermes/quotes/...）。
metadata:
  hermes:
    tags: [stocks, a-share, quotes, watchlist]
---

## 目标

当用户说“把 XXX 加入自选股 / 删除自选股 / 替换一组自选股 / 列出当前自选股”时：
- 必须调用 `quotes_watchlist` 工具完成落盘修改
- 修改后建议再 `list` 一次确认结果，并把最终列表回显给用户

## 工具说明

- `quotes_watchlist(action="list")`：查看自选股列表
- `quotes_watchlist(action="add", code="000001")`：新增
- `quotes_watchlist(action="remove", code="000001")`：删除
- `quotes_watchlist(action="replace", watchlist=[...])`：整体替换（用于批量维护/同步）
- `quotes_watchlist(action="get_prefs")`：查看偏好（包含刷新设置）
- `quotes_watchlist(action="set_prefs", auto_refresh=true, refresh_interval_seconds=30)`：更新刷新设置

## 规范（必须遵守）

1) **代码校验**：股票代码必须是 6 位数字（如 `000001`）。不符合就让用户给正确代码。
2) **去重**：批量替换时要去重并保持用户给的顺序。
3) **不瞎编**：工具调用失败要把 error 原样反馈，并给出下一步排查（例如检查 `HERMES_WEB_UI_BASE` 是否可访问）。

## 示例

### 增加自选股

- 用户：把 300750 加入自选股
- 你：
  1) `quotes_watchlist(add, 300750)`
  2) `quotes_watchlist(list)` 确认

### 删除自选股

- 用户：删掉 002793
- 你：
  1) `quotes_watchlist(remove, 002793)`
  2) `quotes_watchlist(list)` 确认

### 批量替换

- 用户：自选股换成 000001、600519、300750
- 你：
  1) `quotes_watchlist(replace, [000001,600519,300750])`
  2) `quotes_watchlist(list)` 确认

