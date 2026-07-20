#!/usr/bin/env python3
"""业务验证脚本 - 实际调用核心功能验证业务逻辑"""

import sys
import json
import tempfile
from pathlib import Path

NARRATO_AI_PATH = Path.home() / "Downloads/workspace/NarratoAI"
sys.path.insert(0, str(NARRATO_AI_PATH))
NARRATO_VENV = NARRATO_AI_PATH / ".venv/lib/python3.12/site-packages"
if NARRATO_VENV.exists():
    sys.path.insert(0, str(NARRATO_VENV))

AIWRITE_X_PATH = Path.home() / "Downloads/workspace/AIWriteX"
sys.path.insert(0, str(AIWRITE_X_PATH))
AIWRITE_VENV = AIWRITE_X_PATH / ".venv/lib/python3.12/site-packages"
if AIWRITE_VENV.exists():
    sys.path.insert(0, str(AIWRITE_VENV))

NARRATO_ADAPTER = Path.home() / "Downloads/workspace/hermes-agent/optional-skills/narrato-ai"
AIWRITE_ADAPTER = Path.home() / "Downloads/workspace/hermes-agent/optional-skills/aiwrite-x"

import importlib.util

def load_adapter(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

narrato_adapter = load_adapter("narrato_adapter", str(NARRATO_ADAPTER / "adapter.py"))
aiwrite_adapter = load_adapter("aiwrite_adapter", str(AIWRITE_ADAPTER / "adapter.py"))


def test_narrato_tts():
    """业务验证: NarratoAI TTS 配音"""
    print("=" * 60)
    print("业务验证 1: NarratoAI TTS 配音")
    print("=" * 60)
    
    try:
        result = narrato_adapter.generate_tts_audio(
            text="你好，这是一段测试文本，用于验证 TTS 配音功能是否正常工作。",
            voice_name="zh-CN-XiaoxiaoNeural",
            output_path=str(Path(tempfile.mkdtemp()) / "test_tts.mp3")
        )
        
        if result.get("success"):
            audio_path = result.get("audio_path", "")
            if Path(audio_path).exists():
                size = Path(audio_path).stat().st_size
                print(f"✓ TTS 生成成功: {audio_path}")
                print(f"✓ 文件大小: {size / 1024:.1f} KB")
                if size > 1000:
                    print("✓ 文件大小合理（>1KB）")
                    return True
                else:
                    print("✗ 文件过小，可能为空")
                    return False
            else:
                print(f"✗ 音频文件不存在: {audio_path}")
                return False
        else:
            print(f"✗ TTS 生成失败: {result.get('error', 'unknown')}")
            return False
    except Exception as e:
        print(f"✗ TTS 异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_narrato_voice_list():
    """业务验证: NarratoAI 语音列表"""
    print("\n" + "=" * 60)
    print("业务验证 2: NarratoAI 语音列表")
    print("=" * 60)
    
    try:
        voices = narrato_adapter.get_available_voices()
        if len(voices) >= 5:
            print(f"✓ 获取到 {len(voices)} 个语音:")
            for v in voices[:5]:
                print(f"  - {v}")
            return True
        else:
            print(f"✗ 语音数量不足: {len(voices)}")
            return False
    except Exception as e:
        print(f"✗ 语音列表异常: {e}")
        return False


def test_aiwrite_platform_list():
    """业务验证: AIWriteX 平台列表"""
    print("\n" + "=" * 60)
    print("业务验证 3: AIWriteX 平台列表")
    print("=" * 60)
    
    try:
        platforms = aiwrite_adapter.get_supported_platforms()
        expected = ["wechat", "xiaohongshu", "douyin", "toutiao", "baijiahao", "zhihu", "douban"]
        if len(platforms) == 7:
            print(f"✓ 获取到 {len(platforms)} 个平台:")
            for p in platforms:
                status = "✓" if p in expected else "?"
                print(f"  {status} {p}")
            return True
        else:
            print(f"✗ 平台数量不符: {len(platforms)} (期望 7)")
            return False
    except Exception as e:
        print(f"✗ 平台列表异常: {e}")
        return False


def test_narrato_video_gen_validation():
    """业务验证: NarratoAI 视频生成参数校验"""
    print("\n" + "=" * 60)
    print("业务验证 4: NarratoAI 视频生成参数校验")
    print("=" * 60)
    
    try:
        result = narrato_adapter.generate_narration_video(
            video_path="/nonexistent/video.mp4",
            script_json='[{"text": "test", "start": 0, "end": 1}]'
        )
        
        if not result.get("success"):
            error = result.get("message", result.get("error", ""))
            print(f"✓ 参数校验正确（视频文件不存在时正确返回失败）: {error[:100]}")
            return True
        else:
            print("✗ 不应该成功（视频文件不存在）")
            return False
    except Exception as e:
        print(f"✗ 参数校验异常: {e}")
        return False


def test_aiwrite_article_gen_validation():
    """业务验证: AIWriteX 文章生成参数校验"""
    print("\n" + "=" * 60)
    print("业务验证 5: AIWriteX 文章生成参数校验")
    print("=" * 60)
    
    try:
        result = aiwrite_adapter.generate_article(
            topic="",
            platform="invalid_platform"
        )
        
        if not result.get("success"):
            error = result.get("error", "")
            print(f"✓ 参数校验正确: {error}")
            return True
        else:
            print("✗ 不应该成功（平台无效）")
            return False
    except Exception as e:
        print(f"✗ 参数校验异常: {e}")
        return False


def test_concurrent_api_calls():
    """业务验证: 并发 API 调用"""
    print("\n" + "=" * 60)
    print("业务验证 6: 并发 API 调用")
    print("=" * 60)
    
    try:
        import concurrent.futures
        
        def call_narrato():
            return narrato_adapter.check_dependencies()
        
        def call_aiwrite():
            return aiwrite_adapter.check_dependencies()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(call_narrato),
                executor.submit(call_aiwrite),
                executor.submit(call_narrato),
                executor.submit(call_aiwrite),
            ]
            results = [f.result() for f in futures]
        
        all_ok = all(r.get("ok") for r in results)
        if all_ok:
            print(f"✓ 4 个并发调用全部成功")
            return True
        else:
            print(f"✗ 部分并发调用失败")
            return False
    except Exception as e:
        print(f"✗ 并发调用异常: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("业务验证")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("NarratoAI TTS 配音", test_narrato_tts()))
    results.append(("NarratoAI 语音列表", test_narrato_voice_list()))
    results.append(("AIWriteX 平台列表", test_aiwrite_platform_list()))
    results.append(("NarratoAI 参数校验", test_narrato_video_gen_validation()))
    results.append(("AIWriteX 参数校验", test_aiwrite_article_gen_validation()))
    results.append(("并发 API 调用", test_concurrent_api_calls()))
    
    print("\n" + "=" * 60)
    print("业务验证报告")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 验证通过")
    
    if passed == total:
        print("\n✓ 所有业务验证通过！")
        return 0
    else:
        print(f"\n✗ {total - passed} 个验证失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
