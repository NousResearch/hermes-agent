import { AssistantRuntimeProvider, ThreadPrimitive, useExternalStoreRuntime, type ThreadMessageLike, type TextMessagePartProps } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { afterEach, expect, it } from 'vitest'
import { StructuralMessageParts } from './structural-message-parts'

afterEach(cleanup)
let mounts = 0
const Text = ({ text }: TextMessagePartProps) => {
  useEffect(() => { mounts++ }, [])
  return <span>{text}</span>
}
const components = { Text }
const Message = () => <StructuralMessageParts components={components} />
const threadComponents = { Message }
let update: (message: ThreadMessageLike) => void
const msg = (texts: string[]): ThreadMessageLike => ({ id: 'reply', role: 'assistant', content: texts.map(text => ({ type: 'text', text })) })
function Harness() {
  const [message, setMessage] = useState(msg(['intro', 'final']))
  update = setMessage
  const runtime = useExternalStoreRuntime({ messages: [message], convertMessage: m => m, onNew: async () => {} })
  return <AssistantRuntimeProvider runtime={runtime}><ThreadPrimitive.Messages components={threadComponents} /></AssistantRuntimeProvider>
}
it('rebuilds child scopes on shrink, not on token ticks', async () => {
  mounts = 0
  render(<Harness />)
  expect(mounts).toBe(2)
  for (let i = 0; i < 20; i++) {
    await act(async () => update(msg(['intro', `final ${i}`])))
  }
  expect(mounts).toBe(2)
  await act(async () => update(msg(['final answer'])))
  expect(screen.getAllByText('final answer')).toHaveLength(1)
  // A fresh provider cannot retain an accessor from the previous parts shape.
  expect(mounts).toBe(3)
  await act(async () => update(msg(['intro restored', 'final answer'])))
  expect(screen.getAllByText('final answer')).toHaveLength(1)
  expect(mounts).toBe(5)
})
