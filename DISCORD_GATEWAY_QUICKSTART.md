# Hermes Discord Gateway Quickstart

這份小抄是給平常使用看的。

目前你的 Hermes 設定是：

- Discord gateway
- 目前預設主模型：`openrouter:openrouter/free`
- 保留可切回的雲端模型：`openrouter:openrouter/free`
- 額外可切換的本機模型 provider：`ollama-local`
- 額外可切換的 WSL vLLM provider：`vllm-local`

## 1. 進入專案

每次操作前，先進入專案並啟用環境：

```bash
cd /home/dachen/.hermes/hermes-agent
source venv/bin/activate
```

## 2. 啟動 Hermes Discord

建議用背景模式啟動：

```bash
setsid bash -lc 'source /home/dachen/.hermes/hermes-agent/venv/bin/activate && hermes gateway run >> /home/dachen/.hermes/logs/gateway.log 2>&1' >/dev/null 2>&1 < /dev/null &
```

啟動後可再查一次狀態：

```bash
hermes gateway status
```

## 3. 確認 Hermes 有沒有在跑

```bash
cd /home/dachen/.hermes/hermes-agent
source venv/bin/activate
hermes gateway status
```

正常會看到類似：

```text
✓ Gateway is running
```

## 4. 關掉 Hermes Discord

```bash
pkill -f "hermes gateway run"
```

關掉後可以再查一次：

```bash
hermes gateway status
```

## 5. 看 log

看最近 50 行：

```bash
tail -n 50 ~/.hermes/logs/gateway.log
```

持續追 log：

```bash
tail -f ~/.hermes/logs/gateway.log
```

## 6. 在 Discord 怎麼用

- 私訊 bot：直接傳訊息就可以
- Server 頻道：預設要先 `@` bot，它才會回

測試範例：

```text
@hermes 你好
```

或私訊：

```text
你好，你現在用的是什麼模型？
```

### 模型切換小抄

在 Discord 裡直接用 `/model ...` 切換。

建議優先使用這些：

```text
/model openrouter:openrouter/free
/model llama3.2:latest --provider ollama-local
/model qwen3-14b-64k:latest --provider ollama-local
/model qwen2.5:7b --provider ollama-local
/model qwen3.5:9b --provider vllm-local
```

說明：

- `openrouter:openrouter/free`：切回雲端免費模型
- `llama3.2:latest --provider ollama-local`：目前最推薦的快速本機主力，已安裝、128K context、支援 tools
- `qwen3-14b-64k:latest --provider ollama-local`：能力較強，但明顯更慢
- `qwen2.5:7b --provider ollama-local`：本機較省資源、比較快
- `qwen3.5:9b --provider vllm-local`：WSL 版 vLLM 路線，預留給較穩定的本機 agent 服務

你現在本機也可切這些：

```text
/model openrouter:openrouter/free
/model llama3.2:latest --provider ollama-local
/model qwen3-14b-64k:latest --provider ollama-local
/model qwen2.5:7b --provider ollama-local
/model qwen3.5:9b --provider vllm-local
```

切完之後建議馬上問一句確認：

```text
你現在是甚麼模型？
```

如果切完怪怪的，先重置 session：

```text
/reset
```

再重新切一次。

### 切到本機 Ollama

目前我幫你掛進 Hermes 的本機 provider 名稱是：

```text
ollama-local
vllm-local
```

在 Discord 裡可以這樣切：

```text
/model llama3.2:latest --provider ollama-local
```

如果你想切到另一顆本機 Ollama：

```text
/model qwen3-14b-64k:latest --provider ollama-local
```

如果你想切到 WSL 上的 vLLM：

```text
/model qwen3.5:9b --provider vllm-local
```

如果你想切回原本的 OpenRouter：

```text
/model openrouter:openrouter/free
```

你目前本機可用或即將可用的 Ollama 模型：

```text
gemma3:12b
qwen3:14b
qwen3-14b-64k
qwen2.5:7b
gemma2:9b
llama3.2:latest
qwen3.5:9b
```

目前狀態：

- 立刻可用的主預設：`openrouter:openrouter/free`
- `gemma3:12b` 已下載完成，但目前不建議當 Hermes 主模型
- 如果要重新切回雲端預設：`/model openrouter:openrouter/free`
- 如果要切到 WSL vLLM：`/model qwen3.5:9b --provider vllm-local`

建議：

- 想切回雲端預設：`/model openrouter:openrouter/free`
- 想要目前 Hermes 最快的穩定本機主力：`/model llama3.2:latest --provider ollama-local`
- 想要較強但較慢：`/model qwen3-14b-64k:latest --provider ollama-local`
- 想要比較省資源、比較快：`/model qwen2.5:7b --provider ollama-local`
- 想用 WSL 上的 vLLM：`/model qwen3.5:9b --provider vllm-local`

## 8. WSL vLLM 小抄

啟動 `qwen3.5:9b` 的 vLLM：

```bash
cd /home/dachen/.hermes/hermes-agent
./scripts/start_vllm_qwen35_9b.sh
```

查狀態：

```bash
cd /home/dachen/.hermes/hermes-agent
./scripts/status_vllm_qwen35_9b.sh
```

停止：

```bash
cd /home/dachen/.hermes/hermes-agent
./scripts/stop_vllm_qwen35_9b.sh
```

## 7. 最短版指令

啟動：

```bash
cd /home/dachen/.hermes/hermes-agent
source venv/bin/activate
setsid bash -lc 'source /home/dachen/.hermes/hermes-agent/venv/bin/activate && hermes gateway run >> /home/dachen/.hermes/logs/gateway.log 2>&1' >/dev/null 2>&1 < /dev/null &
```

查狀態：

```bash
cd /home/dachen/.hermes/hermes-agent
source venv/bin/activate
hermes gateway status
```

關掉：

```bash
pkill -f "hermes gateway run"
```
