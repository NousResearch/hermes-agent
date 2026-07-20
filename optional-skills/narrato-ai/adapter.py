"""
NarratoAI Adapter for Hermes Agent

深度集成 NarratoAI 影视解说视频生成功能。
通过直接导入 NarratoAI 模块实现零 IPC 开销调用。
"""

import sys
import os
import json
import uuid
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# NarratoAI 项目路径
NARRATO_AI_PATH = os.environ.get(
    "NARRATO_AI_PATH",
    str(Path.home() / "Downloads" / "workspace" / "NarratoAI"),
)

# 将 NarratoAI 加入 sys.path（延迟导入）
_narrato_initialized = False


def _ensure_narrato_path():
    """确保 NarratoAI 路径在 sys.path 中"""
    global _narrato_initialized
    if _narrato_initialized:
        return

    narrato_path = str(Path(NARRATO_AI_PATH).resolve())
    if narrato_path not in sys.path:
        sys.path.insert(0, narrato_path)

    # 激活 NarratoAI 的虚拟环境（如果从 Hermes 环境调用）
    venv_site = Path(NARRATO_AI_PATH) / ".venv" / "lib"
    if venv_site.exists():
        for py_dir in venv_site.iterdir():
            site_packages = py_dir / "site-packages"
            if site_packages.exists() and str(site_packages) not in sys.path:
                sys.path.insert(0, str(site_packages))

    _narrato_initialized = True
    logger.info(f"NarratoAI adapter initialized, path: {narrato_path}")


def _import_narrato_modules():
    """延迟导入 NarratoAI 模块"""
    _ensure_narrato_path()

    from app.models.schema import VideoClipParams, VideoAspect
    from app.services import task as narrato_task

    return VideoClipParams, VideoAspect, narrato_task


def generate_narration_video(
    video_path: str,
    script_json: str,
    voice_name: str = "zh-CN-YunjianNeural",
    video_aspect: str = "9:16",
    tts_engine: str = "edge_tts",
    bgm_name: str = "random",
    output_dir: Optional[str] = None,
    n_threads: int = 8,
    subtitle_enabled: bool = True,
    font_name: str = "SimHei",
    font_size: int = 36,
) -> dict:
    """
    生成完整的影视解说视频。

    Args:
        video_path: 原视频文件路径
        script_json: 解说脚本 JSON 文件路径（或 JSON 字符串）
        voice_name: TTS 语音名称
        video_aspect: 视频比例 (9:16, 16:9, 1:1)
        tts_engine: TTS 引擎 (edge_tts, azure, doubao)
        bgm_name: 背景音乐名称 (random 或具体名称)
        output_dir: 输出目录（默认 NarratoAI/storage/tasks/）
        n_threads: 处理线程数
        subtitle_enabled: 是否启用字幕
        font_name: 字幕字体
        font_size: 字幕字号

    Returns:
        dict: {
            "success": bool,
            "task_id": str,
            "output_path": str,
            "message": str
        }
    """
    try:
        VideoClipParams, VideoAspect, narrato_task = _import_narrato_modules()
    except ImportError as e:
        return {
            "success": False,
            "task_id": "",
            "output_path": "",
            "message": f"NarratoAI 模块导入失败: {e}",
        }

    # 生成任务 ID
    task_id = str(uuid.uuid4())[:8]

    # 解析脚本 JSON
    if os.path.isfile(script_json):
        with open(script_json, "r", encoding="utf-8") as f:
            script_data = json.load(f)
        script_path = script_json
    else:
        # 假设是 JSON 字符串，写入临时文件
        script_data = json.loads(script_json)
        script_path = f"/tmp/narrato_script_{task_id}.json"
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)

    # 映射视频比例
    aspect_map = {
        "9:16": "9:16",
        "16:9": "16:9",
        "1:1": "1:1",
        "3:4": "3:4",
        "4:3": "4:3",
    }
    aspect = aspect_map.get(video_aspect, "9:16")

    # 构建参数
    params = VideoClipParams(
        video_clip_json=script_data,
        video_clip_json_path=script_path,
        video_origin_path=video_path,
        video_origin_paths=[video_path],
        video_aspect=aspect,
        voice_name=voice_name,
        tts_engine=tts_engine,
        bgm_name=bgm_name,
        n_threads=n_threads,
        subtitle_enabled=subtitle_enabled,
        font_name=font_name,
        font_size=font_size,
    )

    try:
        logger.info(f"Starting NarratoAI video generation task: {task_id}")
        narrato_task.start_subclip_unified(task_id, params)

        return {
            "success": True,
            "task_id": task_id,
            "output_path": output_dir or f"storage/tasks/{task_id}",
            "message": f"视频生成任务已启动: {task_id}",
        }
    except Exception as e:
        logger.error(f"NarratoAI video generation failed: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "output_path": "",
            "message": f"视频生成失败: {e}",
        }


def generate_tts_audio(
    text: str,
    voice_name: str = "zh-CN-YunjianNeural",
    tts_engine: str = "edge_tts",
    voice_rate: float = 1.0,
    voice_pitch: float = 1.0,
    output_path: Optional[str] = None,
) -> dict:
    """
    生成 TTS 配音音频。

    Args:
        text: 要转换的文本
        voice_name: 语音名称
        tts_engine: TTS 引擎
        voice_rate: 语速 (0.5-2.0)
        voice_pitch: 语调 (0.5-2.0)
        output_path: 输出音频路径

    Returns:
        dict: {"success": bool, "audio_path": str, "message": str}
    """
    try:
        _ensure_narrato_path()
        from app.services import voice as narrato_voice
    except ImportError as e:
        return {"success": False, "audio_path": "", "message": f"模块导入失败: {e}"}

    task_id = str(uuid.uuid4())[:8]
    out = output_path or f"/tmp/narrato_tts_{task_id}.mp3"

    try:
        segments = [{"narration": text, "OST": 0, "timestamp": "00:00:00", "_id": "0"}]
        results = narrato_voice.tts_multiple(
            task_id=task_id,
            list_script=segments,
            tts_engine=tts_engine,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_pitch=voice_pitch,
        )
        if results and len(results) > 0:
            audio_path = results[0].get("audio_file", "") if isinstance(results[0], dict) else str(results[0])
            return {
                "success": True,
                "audio_path": audio_path,
                "message": "TTS 音频生成成功",
            }
        return {"success": False, "audio_path": "", "message": "TTS 生成无结果"}
    except Exception as e:
        return {"success": False, "audio_path": "", "message": f"TTS 生成失败: {e}"}


def get_available_voices(language: str = "zh") -> list:
    """
    获取可用的 TTS 语音列表。

    Args:
        language: 语言过滤 (zh, en, ja 等)

    Returns:
        list: 可用语音名称列表
    """
    voices = [
        # 中文语音
        "zh-CN-XiaoxiaoNeural",
        "zh-CN-XiaoyiNeural",
        "zh-CN-YunjianNeural",
        "zh-CN-YunxiNeural",
        "zh-CN-YunxiaNeural",
        "zh-CN-YunyangNeural",
        "zh-CN-liaoning-XiaobeiNeural",
        "zh-CN-shaanxi-XiaoniNeural",
        # 英文语音
        "en-US-AnaNeural",
        "en-US-AriaNeural",
        "en-US-AvaNeural",
        "en-US-EmmaNeural",
        "en-US-JennyNeural",
        "en-US-MichelleNeural",
        "en-US-AndrewNeural",
        "en-US-BrianNeural",
        "en-US-ChristopherNeural",
        "en-US-EricNeural",
        "en-US-GuyNeural",
        "en-US-RogerNeural",
        "en-US-SteffanNeural",
    ]

    if language:
        voices = [v for v in voices if v.startswith(language)]

    return voices


def check_dependencies() -> dict:
    """检查 NarratoAI 依赖是否满足。"""
    result = {"ok": True, "issues": []}

    # 检查 FFmpeg
    import shutil
    if not shutil.which("ffmpeg"):
        result["ok"] = False
        result["issues"].append("FFmpeg 未安装，请运行: brew install ffmpeg")

    # 检查 NarratoAI 模块
    try:
        _import_narrato_modules()
    except ImportError as e:
        result["ok"] = False
        result["issues"].append(f"NarratoAI 模块导入失败: {e}")

    return result
