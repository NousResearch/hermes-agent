#Requires -Version 5.1
<#
.SYNOPSIS
  Install Gemma 4 26B via Ollama and point Hermes at the local OpenAI-compatible API.

.NOTES
  - Install Ollama from https://ollama.com/download (Windows) if `ollama` is missing.
  - Hermes docs: https://hermes-agent.nousresearch.com/docs/integrations/providers#ollama--local-models-zero-config
  - Model card: https://ollama.com/library/gemma4 (tag: gemma4:26b)
#>
$ErrorActionPreference = "Stop"
$OllamaBase = "http://127.0.0.1:11434/v1"
$ModelPull = "gemma4:26b"
$ModelHermes = "gemma4-26b-hermes"

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host "未找到 ollama。请从 https://ollama.com/download 安装并重启终端后重试。" -ForegroundColor Yellow
    exit 1
}

# 提高 Ollama 默认上下文（Hermes 工具链需要足够 KV；可按机器改为 32768）
$env:OLLAMA_CONTEXT_LENGTH = if ($env:OLLAMA_CONTEXT_LENGTH) { $env:OLLAMA_CONTEXT_LENGTH } else { "65536" }
Write-Host "OLLAMA_CONTEXT_LENGTH=$($env:OLLAMA_CONTEXT_LENGTH)（仅对本次启动的子进程生效；可在系统环境变量里永久设置）"

Write-Host "→ ollama pull $ModelPull （体积约 18GB，请耐心等待）"
& ollama pull $ModelPull

$repoRoot = Split-Path $PSScriptRoot -Parent
$modelfile = Join-Path $repoRoot "models\Modelfile.gemma4-26b-hermes"
if (Test-Path $modelfile) {
    Write-Host "→ ollama create $ModelHermes （固定 num_ctx 65536）"
    & ollama create $ModelHermes -f $modelfile
    $useModel = $ModelHermes
} else {
    $useModel = $ModelPull
}

Write-Host "→ 写入 Hermes model 配置（provider=custom, base_url=$OllamaBase, default=$useModel）"
python -c @"
from hermes_cli.config import read_raw_config, get_config_path, ensure_hermes_home
import yaml
ensure_hermes_home()
path = get_config_path()
cfg = read_raw_config()
m = cfg.get('model')
if isinstance(m, str) and m.strip():
    m = {'default': m.strip()}
elif not isinstance(m, dict):
    m = {}
m = dict(m)
m['provider'] = 'custom'
m['default'] = '$useModel'
m['base_url'] = '$OllamaBase'
m['context_length'] = 65536
cfg['model'] = m
cfg.setdefault('_config_version', 18)
with open(path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print('Wrote', path)
"@

Write-Host "→ 探测 OpenAI 兼容接口"
curl.exe -s "$OllamaBase/models" | Select-Object -First 1

Write-Host "→ 发一条测试对话（OpenAI SDK）"
python -c @"
from openai import OpenAI
c = OpenAI(base_url='$OllamaBase', api_key='no-key-required')
r = c.chat.completions.create(
    model='$useModel',
    messages=[{'role':'user','content':'Reply with exactly: OK'}],
    max_tokens=32,
)
print(r.choices[0].message.content)
"@

Write-Host "完成。运行 Hermes: `$env:PYTHONIOENCODING='utf-8'; hermes` 或先 `hermes doctor`。"
