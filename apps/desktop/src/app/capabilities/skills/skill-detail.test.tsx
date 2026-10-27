import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { SkillDetail } from './skill-detail'

const getContent = vi.hoisted(() => vi.fn())
vi.mock('@/hermes', () => ({
  getSkillContent: getContent,
  profileScopeKey: (p: string) => p
}))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      skills: {
        edit: 'Edit',
        archive: 'Archive',
        loading: 'Loading',
        noDescription: 'Empty',
        relatedSkills: 'Related skills',
        missingRelatedSkills: 'Missing related skills'
      }
    }
  })
}))
vi.mock('@/components/ui/button', () => ({
  Button: ({ children, size: _size, variant: _variant, ...props }: any) => <button {...props}>{children}</button>
}))
vi.mock('@/components/page-loader', () => ({
  PageLoader: () => <span>Loading</span>
}))
it('keeps the body usable when frontmatter is malformed or absent', async () => {
  getContent.mockResolvedValue({ content: '---\nrelated_skills: [\n---\nReadable body' })
  render(<QueryClientProvider client={new QueryClient()}><SkillDetail skill={{ name: 'bad', category: '', description: '', enabled: true }} skills={[]} onSelectSkill={vi.fn()} onEdit={vi.fn()} onArchive={vi.fn()} /></QueryClientProvider>)
  await screen.findByText('Readable body')
  expect(screen.queryByText('Related skills')).toBeNull()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

it('offers installed declared relations as navigation and keeps missing names non-interactive', async () => {
  getContent.mockResolvedValue({
    content: '---\nrelated_skills: [beta, absent]\n---\nAlpha body'
  })
  const onSelectSkill = vi.fn()
  const props = {
    skill: { name: 'alpha', category: 'test', description: '', enabled: true },
    profile: 'research',
    onArchive: vi.fn(),
    onEdit: vi.fn(),
    skills: [{ name: 'beta', category: 'test', description: '', enabled: false }],
    onSelectSkill
  }
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <SkillDetail {...props} />
    </QueryClientProvider>
  )
  await screen.findByText('Alpha body')
  fireEvent.click(screen.getByRole('button', { name: 'beta' }))
  expect(onSelectSkill).toHaveBeenCalledWith('beta')
  expect(screen.getByText('absent')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'absent' })).toBeNull()
  expect(getContent).toHaveBeenCalledWith('alpha', 'research')
})
