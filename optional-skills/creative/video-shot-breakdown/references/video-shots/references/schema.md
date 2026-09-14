# shots.json 结构

一层：**片 → 镜头（shot）**。没有段、没有场——拉片拆的是成片，成片里只有镜头。

```json
{
  "source": "demo-video.mp4",
  "title": "啥是AI",
  "lang": "zh",
  "meta": { "durationSeconds": 202.9, "fps": 30, "width": 1680, "height": 720, "aspect": "7:3", "codec": "h264", "hasAudio": true },
  "params": { "sceneThreshold": 0.15, "minShotSeconds": 0.3 },
  "seedCuts": [0.9, 1.23, 2.97, "…检测到的全部切点"],
  "manualCuts": [63.5, 127.37],
  "cast": [ { "id": "P1", "name": "老太太", "note": "主角" } ],
  "shots": [ { "…": "见下" } ]
}
```

## 机器字段：模型不许改

这四类字段由 `seed` / `recut` 写入，**改了就等于伪造证据**：

| 字段 | 来源 |
| --- | --- |
| `meta` | ffprobe |
| `seedCuts` | ffmpeg 场景检测的原始切点（含被并掉的碎片，备查） |
| `shot.start` / `shot.end` / `shot.seconds` | 切点相减，两位小数 |
| `shot.motion` | 该区间逐帧差分的中位数（两端剔除，避开切点尖峰） |

要改边界只有一条路：**`recut --split` / `--merge`**。它会重编号、重算时长、重算实测运动，
并把补的刀记进 `manualCuts`。手改必漏一处，门会当场点名。

## shot（镜头）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 镜号 `S01`：两位补零、从 1 起、**按顺序连号**。它是关键帧文件名（`S01a.jpg` / `S01b.jpg`） |
| `start` / `end` | number | 起止秒，两位小数。**相邻镜头首尾相接**，首镜从 0 起，末镜收在片长 |
| `seconds` | number | `end − start`。冗余存一份是为了让门能对账 |
| `motion` | number\|null | 实测帧间变化中位数。短镜可能为 null（采样点不够） |
| `size` | enum | 景别，见 `taxonomy.md`。黑场与字卡用 `none` |
| `category` | enum | 镜头类别——这一镜干什么活 |
| `camera` | enum | 运镜。**和 `motion` 对账**：声称大幅运镜却实测不动，门拦 |
| `transitionIn` | enum | 这一镜**怎么进来**的，可省略（等于 `cut`） |
| `subjects` | string[] | 画内人物的 `cast` 编号；空镜给空数组 |
| `frame` | string | **画面描述**，12 字起，写看得见的东西。空话词表和废话开头都会被拦 |
| `onscreenText` | string | 画面上不是台词的文字：片名、字卡、界面文字。可空 |
| `audio` | string | 台词、旁白、关键音效。**烧录的对白字幕算台词写这里** |
| `rhythm` | enum | **节奏角色**：这一镜为什么留得住人（`hook` / `setup` / `build` / `beat` / `turn` / `payoff` / `breath` / `close`）。**可选，但标了就得整片标全** |
| `rhythmNote` | string | 节奏理由，一句话。写观众这一刻看到什么、为什么不划走。标了 `rhythm` 就必须写 |
| `note` | string | 备注，可选。短于 `minShotSeconds` 的镜头**必须**写（说明是闪切还是检测碎片） |

## params

全部可省略，省略走默认值。按片子调的通常只有前两个。

| 字段 | 默认 | 作用 |
| --- | --- | --- |
| `sceneThreshold` | 0.3 | 场景检测阈值。**暗戏、慢片要往下调**（0.15 左右），快切广告可以往上 |
| `minShotSeconds` | 0.3 | 短于它的碎片在 seed 时并进上一镜 |
| `boundaryTolerance` | 0.05 | 相邻镜头首尾相接的容差 |
| `endTolerance` | 0.25 | 末镜收尾对片长的容差 |
| `cutTolerance` | 0.1 | 镜头边界对齐 `seedCuts` / `manualCuts` 的容差 |
| `staticMaxMotion` | 1.5 | 实测低于它 = 画面几乎没动（运镜门的拦截线） |
| `busyMinMotion` | 12 | 实测高于它 = 动得厉害（只出提示，不拦） |
| `motionGateMinSeconds` | 1 | 短于它的镜头不查运镜（采样点太少，一个尖峰就能翻案） |
| `minFrameChars` | 12 | **中文**画面描述的最低字数 |
| `minFrameWords` | 8 | **英文**画面描述的最低词数。按描述本身的语言选用哪一条 |
| `minRhythmChars` | 8 | 中文节奏理由的最低字数 |
| `minRhythmWords` | 5 | 英文节奏理由的最低词数 |
| `hookWindowSeconds` | 5 | 开篇多少秒内该出现钩子（**只提示不拦**） |
| `flatRun` | 6 | 连续多少镜同一个节奏角色算「节奏平」（**只提示不拦**） |
| `trackHz` | 5 | 运动曲线采样率 |
| `frameDir` | `frames` | 关键帧目录 |

## 运动曲线 track.json（单独一份）

```json
{ "hz": 5, "values": [7.4, 8.9, 14.4, "…每 0.2 秒一个点"] }
```

**不进 shots.json**——它有上千个数字，混在工作稿里只会碍事，还容易被改坏。
`validate` 和 `render` 用 `--track` 挂上；不挂就**明说跳过运镜实测对账**，其余门照跑。

曲线的含义：相邻采样帧缩到 64×36 之后的逐像素差分均值。值越大画面变化越剧烈——
**它不区分是机位在动还是主体在动**，所以门只拦一个方向（见 `taxonomy.md` 的实测档）。

## 关键帧

每镜两张，都由 `frames` 命令抽：

- `frames/S01a.jpg`——镜头**起手**（进入 15% 处，避开转场帧）
- `frames/S01b.jpg`——镜头**收尾**（85% 处）

**a 和 b 对照着看就是运镜**：取景变了是推拉摇移，取景没变只有人动了是固定机位。
`sheet` 把 a 表和 b 表各拼成联系表，一屏看二十几个镜头，比一张张翻快一个数量级。
