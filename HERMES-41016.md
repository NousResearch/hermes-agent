# LINE send_message 問題說明

## 問題摘要
在這台 Linux 機器上，Hermes 透過 `send_message` 傳送 LINE 訊息時，`target="line"` 或 `target="line:U..."` 可能失敗，錯誤表現為：

- `No home channel set for line ...`
- 或回傳錯誤 JSON，而不是成功送出

但 LINE adapter 本身並沒有壞；直接走 LINE adapter 的 standalone push 路徑可以成功送出訊息。

## 根本原因
本次其實有兩條彼此獨立、但會在 LINE 上疊加的問題路徑。

### 問題 A：`send_message` 對 LINE explicit target 解析不完整
根因在 `tools/send_message_tool.py` 的 `_parse_target_ref()`。

原本程式沒有把 LINE 的顯式 recipient ID 視為 explicit target。LINE 常見 recipient 類型為：

- `U...`：user DM
- `C...`：group
- `R...`：room

因為缺少這段判斷，像 `line:Uf24...` 這種 target 會被錯誤地落回：

1. home-channel resolution
2. channel-directory resolution

因此即使 LINE adapter 與 token 都正常，`send_message` 工具層仍然會誤判並報錯。

### 問題 B：新 session / 新 process 對 LINE home channel 的 runtime 載入不完整
補查後確認，LINE home channel 在 plugin 路徑上還有兩個缺口：

1. `plugins/platforms/line/adapter.py::_env_enablement()`
   原本把 `LINE_HOME_CHANNEL` seed 成單純字串，但 `gateway/config.py`
   的 plugin enable pass 期待的是 `{"chat_id": ..., "name": ...}` 形狀，
   否則不會被轉成 `HomeChannel` dataclass。

2. `gateway/config.py`
   built-in 平台有各自的 `*_HOME_CHANNEL` env override 邏輯，但 plugin
   平台沒有通用 home-channel 套用路徑。這導致：
   - `hermes config set LINE_HOME_CHANNEL ...` 寫進 `config.yaml` 的值
     不一定會在新 session 轉回 runtime env
   - 即使 `gateway.platforms.line.enabled: true` 已存在，plugin 平台仍可能
     在 `load_gateway_config()` 後拿不到 `home_channel`

這也是為什麼會出現「同樣是 LINE，explicit target 修了，但新 session 的
`target="line"` 仍然報沒有 home channel」的現象。

## 為什麼 macOS 看起來沒問題
這不是單純的 macOS / Linux 平台差異。

較合理的原因是：

1. macOS 那邊剛好沒有踩到這個 explicit target parsing bug
2. 或 macOS 那邊在 session 啟動前就已經有可用的 LINE home channel 設定
3. 或 macOS 當時使用的是已重啟、已載入新設定/新程式碼的 session

換句話說，這次問題的核心是 Hermes `send_message` 的 LINE target parsing 邏輯，而不是作業系統本身。

## 為什麼修完後當前 session 還是會失敗
Hermes 的 code change 不會自動套用到「已經在跑」的 CLI / gateway process。

Hermes 文件也明確指出：

- tools/skills 變更通常要 `/reset`
- code changes 要 restart CLI 或 gateway process

因此會出現這種狀況：

- repo 裡的 code 已經修好
- `.env` 裡也已有 `LINE_HOME_CHANNEL`
- 但正在執行的 Hermes session 仍然沿用舊邏輯

所以當前 session 還是可能繼續報 `No home channel set for line ...`。

## 解決方式
### 1. 程式修復：explicit target parsing
在 `tools/send_message_tool.py` 新增 LINE explicit target 規則：

- 增加 `_LINE_TARGET_RE = re.compile(r"^\\s*([UCR][A-Za-z0-9]{8,})\\s*$")`
- 在 `_parse_target_ref()` 中補上 `platform_name == "line"` 的分支
- 將符合 `U...` / `C...` / `R...` 的 LINE target 直接回傳為 explicit recipient

這樣 `line:U...` 就不會再掉回 home-channel / channel-directory lookup。

### 2. 程式修復：plugin home channel runtime wiring
這次另外補了兩個 runtime 修正：

- `plugins/platforms/line/adapter.py::_env_enablement()`
  - 將 `LINE_HOME_CHANNEL` seed 成
    `{"chat_id": ..., "name": ...}`
  - 讓 plugin enable pass 能正確轉成 `HomeChannel`

- `gateway/config.py`
  - 新增 plugin 通用的 `*_HOME_CHANNEL` / `*_HOME_CHANNEL_NAME` /
    `*_HOME_CHANNEL_THREAD_ID` YAML→env bridge
  - 在 plugin runtime enable pass 中，根據
    `entry.cron_deliver_env_var` 統一把 home channel 套用到
    `PlatformConfig.home_channel`

這樣新 session / 新 process 在 `target="line"` 路徑下，就能正確讀到
LINE home channel。

### 3. 回歸測試
這次共有兩組 regression tests：

1. `tests/tools/test_send_message_tool.py`
   - `test_line_dm_target_is_explicit`
   - `test_line_dm_target_bypasses_channel_directory`

   目的：
   - 驗證 `line:U...` 會被正確視為 explicit target
   - 驗證它不會再走 channel directory resolution

2. `tests/gateway/test_line_plugin.py`
   - `test_seeds_home_channel_as_dict`
   - `test_top_level_line_home_channel_from_config_yaml_reaches_runtime`

   目的：
   - 驗證 LINE plugin 的 `_env_enablement()` 會回傳正確 `home_channel` 形狀
   - 驗證 `LINE_HOME_CHANNEL` 寫在 `config.yaml` 時，新 runtime 可正確載入

### 4. 執行端生效方式
修完 code 後，需要重啟 Hermes 的執行端，否則正在跑的 session 仍會使用舊邏輯。

可用方式：

- CLI：退出後重新啟動 `hermes`
- CLI session：`/reset`
- gateway：`hermes gateway restart`

## 驗證結果
已執行：

1. `venv/bin/python -m pytest tests/tools/test_send_message_tool.py -q -o addopts=''`
2. `venv/bin/python -m pytest tests/gateway/test_line_plugin.py -q -o addopts=''`
3. 載入 `~/.hermes/.env` 後直接執行 `load_gateway_config()` runtime 檢查

結果：

- `tests/tools/test_send_message_tool.py`：先前已通過（文件原始調查結果）
- `tests/gateway/test_line_plugin.py`：`78 passed`
- 實機 runtime 檢查：
  - `enabled: True`
  - `home_channel.chat_id: Uf24b5c29e5ba2ffac52e5c71564d47dd`
  - `home_channel.name: Home`
  - `home_channel.thread_id: None`

其中我也驗證到一個容易誤判的點：若只是直接跑裸 Python，而沒有先載
`~/.hermes/.env`，process 內看不到 LINE credentials / home channel，會誤以為
runtime 修補無效。真正模擬 Hermes 啟動路徑時，home channel 已可正確解析。

## 受影響與修改檔案
本次相關檔案包含：

- `tools/send_message_tool.py`
- `tests/tools/test_send_message_tool.py`
- `plugins/platforms/line/adapter.py`
- `gateway/config.py`
- `tests/gateway/test_line_plugin.py`
- `HERMES-41016.md`

## PR 資訊
已建立 upstream PR：

- PR #41016
- https://github.com/NousResearch/hermes-agent/pull/41016

## 一句話結論
這次不是 LINE adapter 壞掉，也不是 Linux 特有問題；實際上有兩個疊加 bug：一個是 `send_message` 對 LINE explicit target 的解析缺陷，另一個是 plugin 平台在新 session / 新 process 下沒有把 LINE home channel 正確接進 runtime，而兩者都修完後，`line:U...` 與 `target="line"` 都能走回正確路徑。
