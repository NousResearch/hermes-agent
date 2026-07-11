import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $newChatProfile } from '@/store/profile'

const getAiEmployees = vi.fn()
const updateAiEmployeeMetadata = vi.fn()
const getProfileSoul = vi.fn()
const updateProfileSoul = vi.fn()

vi.mock('@/hermes', () => ({
  getAiEmployees: () => getAiEmployees(),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn(),
  updateAiEmployeeMetadata: (profileId: string, body: Record<string, unknown>) =>
    updateAiEmployeeMetadata(profileId, body),
  getProfileSoul: (name: string) => getProfileSoul(name),
  updateProfileSoul: (name: string, content: string) => updateProfileSoul(name, content)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const employees = [
  {
    profile_id: 'default',
    display_name_zh: '总控主理人',
    role_zh: '最高权限总调度',
    mission_zh: '拆解任务并调度专业 Agent。',
    category: 'orchestrator',
    emoji: '🧭',
    sort_order: 10,
    profile_path: '/tmp/hermes',
    soul_path: '/tmp/hermes/SOUL.md',
    model: 'gpt-5.5',
    provider: 'openai-codex',
    skill_count: 12,
    gateway_running: true,
    description: '总控主理人：拆解任务并调度专业 Agent。',
    system_name_locked: true
  },
  {
    profile_id: 'douyin-ingest-agent',
    display_name_zh: '抖音素材采集员',
    role_zh: '短视频链接解析专家',
    mission_zh: '采集短视频素材。',
    category: 'video-pipeline',
    emoji: '📥',
    sort_order: 20,
    profile_path: '/tmp/hermes/profiles/douyin-ingest-agent',
    soul_path: '/tmp/hermes/profiles/douyin-ingest-agent/SOUL.md',
    model: 'gpt-5.5',
    provider: 'openai-codex',
    skill_count: 3,
    gateway_running: false,
    description: '抖音素材采集员：采集短视频素材。',
    system_name_locked: true
  }
]

function renderPage() {
  return import('./index').then(({ AiEmployeesView }) =>
    render(
      <MemoryRouter initialEntries={['/ai-employees']}>
        <AiEmployeesView />
      </MemoryRouter>
    )
  )
}

beforeEach(() => {
  $newChatProfile.set(null)
  getAiEmployees.mockResolvedValue({ agents: employees })
  getProfileSoul.mockResolvedValue({ exists: true, content: 'Original SOUL' })
  updateAiEmployeeMetadata.mockImplementation((profileId: string, body: Record<string, unknown>) =>
    Promise.resolve({ ...employees.find(e => e.profile_id === profileId), ...body })
  )
  updateProfileSoul.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('AiEmployeesView', () => {
  it('constrains the employee list so overflowing employees remain scrollable', async () => {
    const { container } = await renderPage()

    await screen.findByText('2 个员工')

    const sidebar = container.querySelector('aside')
    const employeeList = sidebar?.querySelector('.overflow-y-auto')

    expect([...sidebar!.classList]).toEqual(expect.arrayContaining(['flex', 'min-h-0', 'flex-col']))
    expect([...employeeList!.classList]).toEqual(expect.arrayContaining(['min-h-0', 'flex-1', 'overflow-y-auto']))
  })

  it('renders Chinese employee names while keeping the stable profile id visible', async () => {
    await renderPage()

    expect(await screen.findByText('AI 员工')).toBeTruthy()
    fireEvent.click(await screen.findByText('抖音素材采集员'))

    expect(screen.getAllByText('douyin-ingest-agent').length).toBeGreaterThan(0)
    expect(screen.getByText('短视频链接解析专家')).toBeTruthy()
  })

  it('saves training metadata and SOUL.md for the selected employee', async () => {
    await renderPage()
    fireEvent.click(await screen.findByText('抖音素材采集员'))
    fireEvent.click(screen.getByRole('button', { name: '训练员工' }))

    fireEvent.change(await screen.findByLabelText('中文显示名'), { target: { value: '短视频采集专家' } })
    fireEvent.change(screen.getByLabelText('岗位'), { target: { value: '短视频素材采集与解析' } })
    fireEvent.change(screen.getByLabelText('任务说明'), { target: { value: '解析链接并提取关键帧。' } })
    fireEvent.change(screen.getByLabelText('SOUL.md'), { target: { value: 'Updated SOUL' } })

    fireEvent.click(screen.getByRole('button', { name: '保存训练' }))

    await waitFor(() =>
      expect(updateAiEmployeeMetadata).toHaveBeenCalledWith('douyin-ingest-agent', {
        display_name_zh: '短视频采集专家',
        role_zh: '短视频素材采集与解析',
        mission_zh: '解析链接并提取关键帧。',
        category: 'video-pipeline',
        emoji: '📥'
      })
    )
    expect(updateProfileSoul).toHaveBeenCalledWith('douyin-ingest-agent', 'Updated SOUL')
  })

  it('starts a new chat scoped to the selected employee profile', async () => {
    await renderPage()
    fireEvent.click(await screen.findByText('抖音素材采集员'))

    fireEvent.click(screen.getByRole('button', { name: '用这个员工新建会话' }))

    expect($newChatProfile.get()).toBe('douyin-ingest-agent')
  })
})
