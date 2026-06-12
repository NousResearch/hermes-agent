import subprocess, json, tempfile, os

# 测试API调用
payload = {
    "model": "Qwen3-235B-A22B-w8a8",
    "messages": [{"role": "user", "content": "你好，请回复一个字"}],
    "max_tokens": 10
}

# 写入payload到临时文件
f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
json.dump(payload, f, ensure_ascii=True)
f.close()
temp_file = f.name

# 构建PowerShell脚本
result_file = temp_file + ".result.json"
ps_script = """[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$headers = @{
    "Content-Type" = "application/json; charset=utf-8"
    "Authorization" = "Bearer sk-dooFBpzVWgrvf32YLPFfq5r63dEYHELlUjMT84KrEH5wG0zN"
}
$body = [System.IO.File]::ReadAllText("TEMP_FILE", [System.Text.Encoding]::UTF8)
try {
    $response = Invoke-WebRequest -Uri "https://ai-pool.evebattery.com/v1/chat/completions" -Method POST -Headers $headers -Body $body -TimeoutSec 30
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText("RESULT_FILE", $response.Content, $utf8)
    exit 0
} catch {
    Write-Error $_.Exception.Message
    exit 1
}""".replace("TEMP_FILE", temp_file.replace("\\", "\\\\")).replace("RESULT_FILE", result_file.replace("\\", "\\\\"))

# 写入PowerShell脚本
psf = tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8-sig')
psf.write(ps_script)
psf.close()

# 执行
r = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', psf.name], capture_output=True, timeout=60)
print("Return code:", r.returncode)
print("Stdout:", r.stdout[:500])
print("Stderr:", r.stderr[:500])

# 检查结果文件
if os.path.exists(result_file):
    with open(result_file, 'rb') as f:
        raw = f.read()
    print("Result file bytes:", raw[:200])
    print("Result file text:", raw.decode('utf-8-sig')[:200])
else:
    print("Result file not found")

# 清理
os.unlink(temp_file)
os.unlink(psf.name)
if os.path.exists(result_file):
    os.unlink(result_file)
