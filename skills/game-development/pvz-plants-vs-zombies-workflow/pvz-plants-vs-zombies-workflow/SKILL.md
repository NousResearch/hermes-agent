---
name: pvz-plants-vs-zombies-workflow
description: PRJ-20260421-001 Plants vs Zombies Python+Pygame项目高速迭代流程 - PRD→实现→推送→构建→发布的完整循环
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Game Development, Python, Pygame, PlantsVsZombies]
    related_skills: [game-studio, dbg-card-game-workflow]
---

# PVZ Python+Pygame 高速迭代工作流

PRJ-20260421-001 Plants vs Zombies，开源克隆项目。
GitHub: YeLuo45/plants-vs-zombies
本地路径: /home/hermes/prj-plants-vs-zombies/

## 何时使用

当boss提出 PvZ 项目迭代需求时加载此技能。

## 项目架构（主分支当前状态）

```
source/
  main.py              # 主循环、状态管理、菜单、暂停
  constants.py         # 常量、配置
  component/
    plant.py           # 植物基类 + create_plant 工厂
    zombie.py          # 僵尸基类 + create_zombie 工厂
    bullet.py          # 子弹类（含 HitParticle, ExplosionEffect, SunParticle, BiteParticle）
    menubar.py         # 底部卡片栏
    map.py             # Grid 格子系统
  state/
    level.py           # Adventure 模式状态机（含 LawnMower）
    endless.py         # Endless 模式
    zen.py             # Zen Garden 模式
    lawnbowling.py     # Bowling 模式
    achievements.py    # 成就系统
    save_system.py     # 存档
```

**注意**: effects.py 和 sound_manager.py 已合并到主分支（merge 后）。视觉特效优先放在 component/effects.py 或 state/level.py 的 class。

## 版本历史

| 版本 | 内容 | Commit |
|------|------|--------|
| V1 | 基础游戏循环 + 植物射击 | - |
| V2 | 视觉增强 A+B+C（WalkingZombieAnimator + 子弹命中粒子 + 植物抖动） | 916ada1 |
| V3 | 屏幕震动(cherry bomb) + 僵尸啃咬粒子 | 72596f6 |
| V4 | 新植物 + 新僵尸 + 冰火Combo + 难度曲线 | - |
| V5 | 并行分支合并（E/F/G 方向） | cd80a20, 387765c, 5bbbafe |

## 关键架构模式

### 1. Singleton Manager 模式（避免循环导入）

Manager 类使用 Singleton 而非模块级函数，避免循环导入：

```python
class LeaderboardManager:
    _instance = None

    def __init__(self):
        if LeaderboardManager._instance is not None:
            return
        LeaderboardManager._instance = self
        self.scores = self._load()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls()
        return cls._instance
```

应用场景：LeaderboardManager, AchievementManager, SaveManager 都用此模式。

### 2. take_damage 返回元组 (dead, shred) 模式

zombie.py 的 `take_damage` 返回二元组，调用方解包：

```python
# zombie.py
def take_damage(self, amount, bullet_type='normal'):
    # ... damage logic ...
    newspaper_destroyed = (self.name == 'newspaper' and self.paper_hp <= 0)
    dead = (self.hp <= 0)
    return (dead, newspaper_destroyed)

# level.py / endless.py 调用处
dead, shred = z.take_damage(damage, bullet.type)
if shred:
    self.particles.append(NewspaperShredEffect(z.x, z.y))
    self.sound_manager.play('explode')
```

优势：避免事件系统或全局状态的循环导入问题。

### 3. Paged Card Panel 模式

menubar.py 分页卡片面板实现：

```python
self.all_cards = list(PLANTS.keys())       # 全部植物
self.cards_per_page = 8
self.card_page = 0

def _recalc_pages(self):
    start = self.card_page * self.cards_per_page
    self.card_list = self.all_cards[start:start + self.cards_per_page]
    self._calc_card_rects()

def _calc_card_rects(self):
    for i, name in enumerate(self.card_list):
        x = LEFT_X + i * SLOT_W
        self.card_rects[name] = pygame.Rect(x, Y, SLOT_W, SLOT_H)

# draw() 中渲染左右箭头 + 页码指示
# get_card_at() 先检查箭头点击，再检查卡片点击
```

### 4. 粒子特效类命名规范

| 效果 | 类名 | 位置 |
|------|------|------|
| 报纸被击毁纸片飞舞 | NewspaperShredEffect | source/component/effects.py |
| 樱桃炸弹爆炸 | CherryBombExplosion | source/state/level.py |
| 南瓜粉碎 | SquashSmashEffect | source/component/effects.py |
| 冰弹碎裂 | IceBlastEffect | source/component/bullet.py |
| 啃咬粒子 | BiteParticle | source/component/bullet.py |

## 迭代流程

每次迭代遵循以下步骤，不等boss确认直接推进：

1. **实现** — 修改源文件
2. **语法检查** — `python3 -m py_compile <files>`
3. **构建exe** — `pyinstaller --name PlantsVsZombies --onefile --windowed main.py`
4. **复制exe** — `cp dist/PlantsVsZombies .`
5. **提交推送** — `git add <files> && git commit -m "V{N}: <desc>" && git push`
6. **更新proposal索引** — 修改 proposal-index.md 中对应条目

## 屏幕震动实现模式

LevelState 添加 shake 状态和 trigger 方法:

```python
# __init__ 中:
self.shake_timer = 0
self.shake_intensity = 0

# trigger_shake 方法:
def trigger_shake(self, intensity, duration):
    self.shake_intensity = intensity
    self.shake_timer = duration

# update() 中 decay:
if self.shake_timer > 0:
    self.shake_timer -= dt

# draw() 中计算偏移:
scroll_x, scroll_y = 0, 0
if self.shake_timer > 0:
    import random
    scroll_x = random.randint(-self.shake_intensity, self.shake_intensity)
    scroll_y = random.randint(-self.shake_intensity, self.shake_intensity)

# Grid 和 Menubar 不震，只有游戏实体震:
# p.draw(surface, scroll_x, scroll_y)
# z.draw(surface, scroll_x, scroll_y)
# b.draw(surface, scroll_x, scroll_y)
# e.draw(surface, scroll_x, scroll_y)
```

每个组件 draw() 签名:

```python
def draw(self, surface, scroll_x=0, scroll_y=0):
    x, y = self.rect.centerx - scroll_x, self.rect.centery - scroll_y
```

## 跨文件事件驱动粒子模式

用于zombie攻击植物时生成啃咬粒子:

```python
# zombie.py __init__:
self.just_bitten = False

# zombie.py update() 攻击时:
self.just_bitten = True

# level.py zombie update 循环后检测:
if z.just_bitten and z.attack_target:
    self.particles.append(BiteParticle(z.attack_target.rect.centerx,
                                      z.attack_target.rect.centery))
    z.just_bitten = False
```

## 核心组件 draw() 签名（已统一）

```python
# plant.py
def draw(self, surface, scroll_x=0, scroll_y=0):

# zombie.py
def draw(self, surface, scroll_x=0, scroll_y=0):

# bullet.py (Bullet, HitParticle, ExplosionEffect, SunParticle, BiteParticle)
def draw(self, surface, scroll_x=0, scroll_y=0):
```

## 分叉历史合并策略（V5 与 origin/master P2-F）

本地 master (V5) 与 origin/master (P2-F) 完全独立开发，共享提交历史。合并方法：

```bash
git merge --allow-unrelated-histories origin/master
```

### 文件级别冲突解决模式

| 文件类型 | 策略 | 方法 |
|----------|------|------|
| constants.py（配置型） | 以某一版本为 base 完全重写 | write_file 合并两者 |
| level.py（大量逻辑） | 逐区域 patch，working tree 可能有意外残留 | grep 确认无冲突标记 |
| bullet.py（多 class） | 用 execute_code 读取两个版本后拼接 | 读 diff 后 write_file |
| plant.py（多 class） | 取 P2 为 base，patch V5 的特定方法 | patch |
| effects.py | 追加 class 而非合并 | append 到 P2 版本末尾 |

### execute_code 合并 bullet.py 示例

```python
import subprocess
v5 = subprocess.run(['git', 'show', 'master:source/component/bullet.py'],
                   capture_output=True, text=True, cwd=path).stdout
p2 = subprocess.run(['git', 'show', 'origin/master:source/component/bullet.py'],
                    capture_output=True, text=True, cwd=path).stdout
# 分析两个版本的 Bullet class 结构
# 取 V5 的 fire/ice + P2 的 splash/IceBlastEffect
# write_file 合并结果
```

### 关键差异（V5 vs P2）

- **V5 Bullet.draw**: `def draw(self, surface, scroll_x=0, scroll_y=0)` — 有 scroll
- **P2 Plant.draw**: `def draw(self, surface)` — 无 scroll
- **V5 PotatoMine**: armed_timer >= 5.0（第 5 秒才炸）
- **P2 PotatoMine**: armed_timer >= 3.0（第 3 秒就炸）
- **V5 effects.py**: SteamEffect（冰+火蒸汽）
- **P2 effects.py**: 无 SteamEffect（只有 ExplosionEffect/IceShatterEffect）

### Rebase 失败后的替代方案

v5-rebase 分支方案失败（rebase 后强制 push 报错），改用 merge 方案。

## 已知坑点

1. **Grid.draw() 不接受 scroll 参数** — Grid 背景固定不动，不参与屏幕震动
2. **Menubar 固定** — 不参与屏幕震动
3. **subagent 改动被 git checkout 还原** — 重要修改不要依赖 subagent，已发生两次
4. **LawnMower** — 定义在 level.py 底部，draw() 不需要 scroll 参数
5. **pyinstaller 构建** — 首次慢，后续增量快；确保所有依赖正确
6. **分叉 histories** — `--allow-unrelated-histories` 后强制合并，冲突只能手动解
7. **rebase 强制 push** — `git push origin v5-rebase --force` 可能因远程历史不符被拒，改用 merge
8. **endless.py 缺少子弹碰撞检测** — level.py 有完整的 bullet→zombie 碰撞循环，但 endless.py 只有 `b.update()` 没有对 zombie 的伤害判定。无尽模式子弹不杀僵尸。需要参考 level.py ~180-200 行的碰撞逻辑补充到 endless.py
9. **take_damage 调用处必须解包** — 所有 `z.take_damage()` 调用处必须用 `dead, shred = z.take_damage(...)` 接收返回值，shred=True 时在调用处生成特效。漏写会导致特效不触发
10. **effects.py 类名与 level.py 重复** — CherryBombExplosion 定义在 level.py 而非 effects.py；新增特效时确认类所在文件

## 构建命令

```bash
cd /home/hermes/prj-plants-vs-zombies
pyinstaller --name PlantsVsZombies --onefile --windowed main.py
cp dist/PlantsVsZombies PlantsVsZombies  # 复制到项目根目录
```

exe 路径: `/home/hermes/prj-plants-vs-zombies/PlantsVsZombies` (~33MB)
