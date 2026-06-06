# Stream Expired 逻辑验证

## 修复后的逻辑

### 场景 1：有 turn_id（GatewayStreamConsumer）

```python
# send_stream_frame(..., turn_id="uuid1")

# Step 1: 入口检查（line 2032-2042）
turn_id = kwargs.get("turn_id")  # "uuid1"
if not turn_id and chat in self._stream_expired_chats:
    return False
# ✅ 有 turn_id，跳过 chat 级别检查

# Step 2: 查找/创建 turn（line 2088-2130）
if turn_id:
    turn_key = f"{chat}:{turn_id}"
    turn = self._stream_turns.get(turn_key)
    if not turn:
        # 创建新 turn（不检查 chat expired）
        turn = StreamTurn(chat, req_id)
        self._stream_turns[turn_key] = turn

# Step 3: 检查特定 turn 是否 expired（line 2132-2134）
if turn.expired:
    return False
# ✅ 只检查这个 turn 的状态

# 结论：即使 chat 在 _stream_expired_chats 中，
# 已有 turn_id 的 turn 可以继续 finalize ✅
```

### 场景 2：无 turn_id（fallback 或直接调用）

```python
# send_stream_frame(...) 无 turn_id

# Step 1: 入口检查
turn_id = kwargs.get("turn_id")  # None
if not turn_id and chat in self._stream_expired_chats:
    return False  # ✅ 阻止新 turn 创建

# Step 2: 查找已有 turn
existing_turn = self._find_active_turn_for_chat(chat)
if existing_turn:
    # 复用已有 turn（可能是旧代码路径）
    turn = existing_turn
else:
    # 创建新 turn
    if chat in self._stream_expired_chats:
        return False  # ✅ 再次检查，阻止新 turn

# 结论：无 turn_id 时，chat expired 会阻止新 turn ✅
```

## 并发场景验证

### 场景：两个并发 consumer

```python
# Consumer 1
consumer1 = GatewayStreamConsumer(turn_id="uuid1")
await send_stream_frame("frame1", turn_id="uuid1")  # 创建 turn1

# Consumer 2
consumer2 = GatewayStreamConsumer(turn_id="uuid2")
await send_stream_frame("frame2", turn_id="uuid2")  # 创建 turn2

# Consumer 1 遇到 stream expired
# → turn1.expired = True
# → _stream_expired_chats.add(chat)

# Consumer 2 继续 finalize
await send_stream_frame("final", finalize=True, turn_id="uuid2")
  # Step 1: turn_id="uuid2" 存在，跳过 chat 级别检查 ✅
  # Step 2: turn_key = "chat:uuid2"，找到 turn2
  # Step 3: turn2.expired? No ✅
  # Step 4: 成功 finalize ✅

# 结论：Consumer 2 不受 Consumer 1 的 expired 影响 ✅
```

### 场景：Consumer 1 expired 后，新的 Consumer 3

```python
# Consumer 3 (新创建)
consumer3 = GatewayStreamConsumer(turn_id="uuid3")
await send_stream_frame("frame3", turn_id="uuid3")
  # Step 1: turn_id="uuid3" 存在，跳过 chat 级别检查
  # Step 2: turn_key = "chat:uuid3"，turn 不存在
  # Step 3: 创建新 turn... 但这里没有检查 chat expired！

# ❌ 问题：turn_id 分支创建新 turn 时不检查 chat expired
```

## 发现的问题

在 turn_id 分支创建新 turn 时，没有检查 `_stream_expired_chats`！

需要修复：

```python
if turn_id:
    turn_key = f"{chat}:{turn_id}"
    turn = self._stream_turns.get(turn_key)
    if not turn:
        # 创建新 turn 前检查 chat 是否 expired
        if chat in self._stream_expired_chats:
            return False  # ← 需要添加
        
        req_id = self._resolve_stream_req_id(chat, reply_to)
        ...
```
