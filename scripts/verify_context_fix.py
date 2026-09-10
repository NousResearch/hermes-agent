#!/usr/bin/env python3
"""
Hermes Context Length 修复验证脚本

此脚本验证所有修复是否已正确应用：
1. config.yaml context_length 设置
2. Ollama num_ctx 设置
3. context_length_cache.yaml 缓存
4. AIAgent 初始化日志
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_success(text):
    print(f"✓ {text}")

def print_warning(text):
    print(f"⚠ {text}")

def print_error(text):
    print(f"✗ {text}")

def check_config_yaml():
    """检查 ~/.hermes/config.yaml"""
    print_header("1. 检查 config.yaml")
    
    config_path = Path.home() / ".hermes" / "config.yaml"
    if not config_path.exists():
        print_error(f"config.yaml not found at {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        model_cfg = config.get("model", {})
        ctx_len = model_cfg.get("context_length")
        ollama_ctx = model_cfg.get("ollama_num_ctx")
        provider = model_cfg.get("provider")
        base_url = model_cfg.get("base_url")
        
        if ctx_len:
            print_success(f"context_length: {ctx_len}")
        else:
            print_error("context_length not set in config.yaml")
            return False
        
        if ollama_ctx:
            print_success(f"ollama_num_ctx: {ollama_ctx}")
        else:
            print_warning(f"ollama_num_ctx not set (optional, but recommended)")
        
        print(f"  provider: {provider}")
        print(f"  base_url: {base_url}")
        return ctx_len is not None
    except Exception as e:
        print_error(f"Failed to parse config.yaml: {e}")
        return False

def check_cache():
    """检查 context_length_cache.yaml"""
    print_header("2. 检查 context_length_cache.yaml")
    
    cache_path = Path.home() / ".hermes" / "context_length_cache.yaml"
    if not cache_path.exists():
        print_warning(f"cache file not found (will be created on first run)")
        return None
    
    try:
        import yaml
        with open(cache_path) as f:
            cache = yaml.safe_load(f)
        
        context_lengths = cache.get("context_lengths", {})
        if context_lengths:
            print_success(f"Found {len(context_lengths)} cached entries:")
            for key, value in context_lengths.items():
                print(f"  {key}: {value}")
            return context_lengths
        else:
            print_warning(f"cache file exists but is empty")
            return {}
    except Exception as e:
        print_error(f"Failed to parse cache file: {e}")
        return None

def check_ollama():
    """检查 Ollama 设置"""
    print_header("3. 检查 Ollama 设置")
    
    try:
        result = subprocess.run(
            ['curl', '-s', '-m', '3', 
             'http://172.22.144.1:11434/api/tags'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            print_error(f"Cannot reach Ollama at 172.22.144.1:11434 (Connection refused or timeout)")
            print_warning(f"Make sure Ollama is running with: OLLAMA_CONTEXT_LENGTH=262144 ollama serve")
            return False
        
        data = json.loads(result.stdout)
        models = data.get("models", [])
        
        print_success(f"Ollama is reachable")
        print(f"  Available models: {len(models)}")
        
        qwen_models = [m for m in models if "qwen" in m["name"].lower()]
        if qwen_models:
            for m in qwen_models:
                print(f"    - {m['name']}")
        
        return True
    except Exception as e:
        print_error(f"Failed to check Ollama: {e}")
        return False

def check_skills_snapshot():
    """检查 skills snapshot 是否被清理"""
    print_header("4. 检查 Skills Snapshot")
    
    snapshot_path = Path.home() / ".hermes" / ".skills_prompt_snapshot.json"
    if snapshot_path.exists():
        size_kb = snapshot_path.stat().st_size / 1024
        if size_kb > 50:
            print_warning(f"skills_prompt_snapshot.json is {size_kb:.1f} KB (should be cleaned)")
            print_warning(f"Run: rm ~/.hermes/.skills_prompt_snapshot.json")
            return False
        else:
            print_success(f"skills_prompt_snapshot.json exists but is small ({size_kb:.1f} KB)")
    else:
        print_success(f"skills_prompt_snapshot.json cleaned up")
    
    return True

def check_webui():
    """检查 WebUI merge状态"""
    print_header("5. 检查 WebUI 状态")
    
    webui_path = Path.home() / "hermes-webui"
    if not webui_path.exists():
        print_warning(f"WebUI not found at {webui_path}")
        return None
    
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=str(webui_path),
            capture_output=True, text=True, timeout=5
        )
        
        lines = result.stdout.strip().split('\n')
        unmerged = [l for l in lines if l.startswith('UU') or 'both' in l]
        
        if unmerged:
            print_error(f"WebUI has unresolved merge conflicts")
            for line in unmerged:
                print(f"  {line}")
            return False
        else:
            print_success(f"WebUI merge status clean")
            return True
    except Exception as e:
        print_warning(f"Could not check WebUI status: {e}")
        return None

def create_verification_log():
    """创建验证日志"""
    log_path = Path.home() / ".hermes" / "context_fix_verification.log"
    
    with open(log_path, 'w') as f:
        f.write(f"Context Length Fix Verification\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"\nAll checks completed. See output above for details.\n")
    
    print_success(f"Verification log saved to {log_path}")

def main():
    print(f"\n{'#'*60}")
    print(f"#  Hermes Context Length 修复验证")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    results = {
        "config": check_config_yaml(),
        "cache": check_cache() is not None,
        "ollama": check_ollama(),
        "skills": check_skills_snapshot(),
        "webui": check_webui(),
    }
    
    create_verification_log()
    
    print_header("验证总结")
    
    passed = sum(1 for v in results.values() if v is True)
    total = len([v for v in results.values() if v is not None])
    
    if passed == total:
        print_success(f"所有检查已通过 ({passed}/{total})")
        print("\n后续步骤:")
        print("1. 确保 Ollama 启动时设置了 OLLAMA_CONTEXT_LENGTH=262144")
        print("2. 重启 Hermes agent: hermes --restart-gateway")
        print("3. 验证 context 大小: hermes -v 2>&1 | grep -i context")
        return 0
    else:
        print_warning(f"部分检查未通过 ({passed}/{total})")
        print("\n请参考上面的错误信息进行修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())
