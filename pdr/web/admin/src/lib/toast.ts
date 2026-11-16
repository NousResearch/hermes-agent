import { ref } from 'vue'

/** 全局轻提示：操作结果用一行文字告知，2.4 秒后消失 */
export const toastText = ref('')
let timer: ReturnType<typeof setTimeout> | undefined

export function toast(text: string): void {
  toastText.value = text
  clearTimeout(timer)
  timer = setTimeout(() => (toastText.value = ''), 2400)
}

/** 把接口错误转成可读提示（403 等后端校验失败也走这里） */
export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : '操作失败，请稍后再试'
}
