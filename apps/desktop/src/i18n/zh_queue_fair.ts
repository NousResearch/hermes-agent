import type { QueueFairCopy } from './types_queue_fair'

export const zhQueueFair: QueueFairCopy = {
  holdUnknown: reason => `等待中：${reason}`,
  hold: {
    session_busy: '正在等待当前轮次结束',
    slot: '正在等待空闲会话名额',
    editing: '正在编辑，暂不开始',
    restarting: '正在等待网关重启',
    paused: '已暂停，直到丢失的轮次被处理',
    provider_limit: '正在等待服务商限额重置',
    memory: '正在等待可用内存'
  },
  moveUp: '上移',
  moveDown: '下移',
  holdFailed: '编辑期间无法暂停这条排队消息；它可能在保存前开始'
}
