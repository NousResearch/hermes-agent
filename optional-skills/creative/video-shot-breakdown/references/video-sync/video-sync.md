## video-sync

把 **shots.json + 原片** 合成一条视频：画面一边，**当前镜头的分镜信息**另一边，
镜头切了信息跟着切，镜头表自动滚动并高亮。

**版式只看宽高比，不看别的：**

| 原片 | 版式 | 合成方式 |
| --- | --- | --- |
| 横版（宽 > 高）、方版 | **画面在上，信息在下** | `vstack` |
| 竖版（高 > 宽） | **画面在左，信息在右** | `hstack` |

两个方向都保证**画面原样缩放、不裁不拉**，面板补足剩下的地方。

**只截三张图，动的部分交给 ffmpeg。** 面板上会变的只有两样：滚到哪、哪一行亮。
所以量一次行位置、截一张底板、截两张把镜头表铺开的长图（一张压暗、一张全亮），
滚动＝从暗底长图裁一个视窗，高亮＝从亮条长图裁一行盖上去，位置由 `sendcmd`
在每个切点下发一条小表达式驱动。53 镜的片子截图从 53 次降到 3 次（12 秒）。

**当前镜头钉在第二行**：第 1、2 镜不滚（高亮往下挪一行），第 3 镜起每切一次往上滚一行。
**动在切点，静在镜头里**——切点处滚 `--ease`（默认 0.45）秒到位就停住，
16 秒的长镜头里让列表一直爬，观众要盯着它晃 16 秒。

**行高由内容决定**，描述写得长行就高；高亮条按行高分层（`crop` 的高度运行时改不动）。

`{baseDir}` = 本文件所在目录。脚本 `{baseDir}/scripts/video-sync.mjs`，零依赖，`node` 直接跑。

**边界（不做的事）**：不做拉片（那是 `video-shots` 的活，本 skill 吃它的 shots.json）、
不剪辑不转场不配乐、不烧字幕到画面上、不做逐帧渲染（动效由 ffmpeg 按时间裁窗做出来）。

---

### Step 0 — 先有拉片数据

要一份 `shots.json`（`video-shots` 的产出）和**对应的原片**。有 `frames/`（每镜关键帧）
更好——镜头表里会用起手帧当缩略图，没有就留空格。

**shots.json 必须和这条片子对得上**：`meta.durationSeconds` 与原片不符就别合，
先回去把拉片做对。

### Step 1 — 先看几何，再动视频

```bash
node {baseDir}/scripts/video-sync.mjs plan shots.json --video <片>
```

只算不合成，几秒出结果：版式、画面区、面板区、输出尺寸。**先确认这三个数字合理再往下走**——
合成一条 3 分钟的片子要一两分钟，算错了再重来不值。

不满意就调：`--panel <比例>`（横版是面板高÷画面高，默认 0.8；竖版是面板宽÷画面宽，默认 1.6）、
`--width` / `--height`（画面区的上限，默认 1920 / 1080）。

**原片太小就放大画面区**：面板是跟着画面区算的，640×360 的片子面板只有 640×288，
四列挤在一起读不了。`--scale 2` 把画面区放到 1280×720、面板放到 1280×576，宽高比不动。
默认是 1（只缩不放）。`plan` / `panels` / `compose` / `export` 都要带上同一个 `--scale`。

### Step 2 — 渲面板

```bash
node {baseDir}/scripts/video-sync.mjs panels shots.json --video <片> \
  --frames frames --lang zh|en [--out panels] [--chrome <路径>]
```

产出 `static.png`（底板）、`list-dim.png` / `list-lit.png`（整张镜头表铺开的长图，
一暗一亮）、`layout.json`（每行的位置，合成时按它算滚动），外加一张 `panels/panel.html`——
**用浏览器打开它就能预览面板**，`#S07` 换镜头看效果（预览里的滚动动效和导出一致）。

**改布局就改 `{baseDir}/scripts/panel.css`**，改完重跑 `panels` 就是新版面，
不用碰脚本、不用重新想截图逻辑。样式约定见 `{baseDir}/references/panel-style.md`。

### Step 3 — 合成

```bash
node {baseDir}/scripts/video-sync.mjs compose shots.json --video <片> \
  [--panels panels] [-o out.mp4] [--crf 20] [--ease 0.45]
```

音轨照搬原片（无声片自动不接音轨）。合成完 stderr 会报输出尺寸、时长、版式，**核对一眼**。

两步也可以一条龙：

```bash
node {baseDir}/scripts/video-sync.mjs export shots.json --video <片> -o out.mp4 --frames frames
```

### Step 4 — 抽帧验收 ⛔ 不能跳

**不要只看合成没报错就交。** 至少抽三帧看看信息对不对得上画面：

```bash
for t in 5 30 60; do ffmpeg -v error -y -ss $t -i out.mp4 -frames:v 1 -q:v 3 /tmp/check-$t.jpg; done
```

看三件事：**这一帧的画面属于哪一镜、面板上的镜号是不是同一个、高亮行有没有跟着走**。
对不上通常是 shots.json 和原片不是同一条，或者 `--video` 指错了。

汇报一句话说清：输出路径、尺寸、时长、版式、多少镜、面板哪几张没渲出来（如果有）。

---

## 边界

- **滚动只发生在切点**：滚 `--ease` 秒到位就停住，镜头里面板是静止的。
  真要做「全程匀速爬」也行（改 `motionPlan` 的分段），但长镜头会一直晃眼睛
- **需要一个无头浏览器**（Chrome / Chromium / Edge）。这是「布局能随便改」的代价：
  换成 ffmpeg 的 `drawtext` 画面板，中文换行、缩略图、高亮全得自己实现，改个版式要改滤镜图
- 词表（景别 / 类别 / 运镜 / 转场）在本 skill 里**自带一份**，不跨目录 import——
  skill 要能整个拷走。`video-shots` 那边加了新词，这边也要加，不然显示成枚举键
- 渲面板固定三张截图，和镜头数无关：53 镜约 12 秒。合成按片长走，203 秒的片子约 75 秒

## 自测

```bash
node {baseDir}/scripts/selftest.mjs
```

122 项断言，不碰 ffmpeg、不开浏览器：几何（横/方/竖、偶数边长、上下限、旋钮）、
面板页面的数据契约（词表下发、节奏角色、缩略图有没有才给、转义）、
动画命令（每条都必须短——表达式长了 ffmpeg 直接配置失败）、ffmpeg 参数。
改完脚本先跑这个。
