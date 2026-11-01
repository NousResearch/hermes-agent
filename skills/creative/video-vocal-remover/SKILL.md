---
name: video-vocal-remover
description: 从视频中去除人声但保留机械声/环境声。当用户要求去掉视频里的人声、消除说话声、保留机器声、视频静音但保留环境音、去除语音保留噪声时使用。也适用于用户说"视频里有说话声想去掉"、"只保留机械声音"、"去掉录音里的人声"等场景。不要用于完全去除所有音频（那是去音轨，用 ffmpeg -an 即可），此技能专门用于频率分离——削弱人声频段而保留其他声音。
---

# 视频人声去除技能

从视频文件中削弱人声频段（300Hz–3400Hz），同时保留低频机械声（如电机、齿轮运动声）和高频环境声（如碰撞、摩擦噪声）。

## 原理

人声的能量集中在 300Hz–3400Hz 频段，而机器人/机械运动的声音主要分布在低于 300Hz 的低频段和高于 3400Hz 的高频段。通过 ffmpeg 的 equalizer 滤波器对这些中频段做大幅衰减，可以有效削弱人声，保留机械声。

## 前置条件

需要 ffmpeg。如果未安装：

```bash
# Ubuntu/Debian
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Conda
conda install -y ffmpeg
```

## 处理单个视频

```bash
ffmpeg -i "INPUT.mp4" \
  -c:v copy \
  -af "equalizer=f=500:t=q:w=1:g=-25,equalizer=f=1000:t=q:w=1:g=-25,equalizer=f=2000:t=q:w=1:g=-25,equalizer=f=3000:t=q:w=1:g=-20" \
  -c:a aac -b:a 128k \
  -y "INPUT_novoice.mp4"
```

参数说明：
- `-c:v copy` — 视频流直接复制，不重新编码，速度极快
- `-af` — 音频滤波链，4 个 equalizer 分别在 500/1000/2000/3000Hz 做 -25dB/-20dB 衰减
- `t=q` — 滤波器类型为 Q 值（带宽参数）
- `w=1` — 带宽系数，1 表示中等宽度
- `g=-25` / `g=-20` — 衰减增益，-25dB 对 500-2000Hz（人声核心频段），-20dB 对 3000Hz（人声高频边缘，衰减稍弱以避免过度影响高频机械声）
- `-c:a aac -b:a 128k` — 音频重新编码为 AAC 128kbps

## 批量处理文件夹内所有视频

对指定文件夹下所有 mp4 文件（不含已处理的 `_novoice` 文件）批量去除人声：

```bash
find "FOLDER_PATH" -type f -name "*.mp4" ! -name "*_novoice.mp4" -print0 | while IFS= read -r -d '' f; do
  echo "Processing: $f"
  ffmpeg -i "$f" \
    -c:v copy \
    -af "equalizer=f=500:t=q:w=1:g=-25,equalizer=f=1000:t=q:w=1:g=-25,equalizer=f=2000:t=q:w=1:g=-25,equalizer=f=3000:t=q:w=1:g=-20" \
    -c:a aac -b:a 128k \
    -y "${f%.mp4}_novoice.mp4" 2>&1 | tail -3
  echo "Done: ${f%.mp4}_novoice.mp4"
done
```

也支持其他视频格式（avi, mkv, mov, webm 等），修改 `find` 的 `-name` 匹配规则即可。

## 调整参数

如果效果不理想，可以根据实际情况微调：

| 场景 | 调整方式 |
|---|---|
| 人声仍然明显 | 增大衰减值，如 g=-30 或 g=-35 |
| 机械声也被削弱了 | 降低衰减值，如 g=-15 或 g=-20；或减小带宽 w=0.5 |
| 只想去除说话声保留所有其他声 | 增大带宽 w=2 让滤波更精准集中在人声核心频段 |
| 视频里人声很低沉（男声为主） | 增加低频衰减：加一个 `equalizer=f=300:t=q:w=1:g=-20` |
| 视频里人声很尖锐（女声/儿童为主） | 增加高频衰减：加一个 `equalizer=f=3500:t=q:w=1:g=-20` |

## 输出文件命名

输出文件在原始文件名后追加 `_novoice`，放在同一目录下。例如：
- `抓取测试-1.mp4` → `抓取测试-1_novoice.mp4`

如果用户要求原地替换（覆盖原文件），先生成 `_novoice` 文件确认效果后，再执行替换：

```bash
mv "INPUT_novoice.mp4" "INPUT.mp4"
```

## 局限性

- 这是基于频率滤波的方法，不是 AI 人声分离。如果人声和机械声在同一频段重叠，无法完全分离。
- 对于背景音乐中的低频人声（如哼唱），效果可能有限。
- 如果需要更高质量的人声分离，可以考虑使用 demucs 等 AI 模型，但那需要 Python 环境和 GPU。