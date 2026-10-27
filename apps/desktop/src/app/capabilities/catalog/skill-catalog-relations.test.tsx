import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { queryClient } from '@/lib/query-client'
import { SkillCatalog } from './skill-catalog'
import { SkillDetail } from '../skills/skill-detail'
import { $catalogCardView } from './store'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof import('@/hermes')>()),
  getOfficialSkills: vi.fn().mockResolvedValue({ skills: [] }),
  getSkillContent: vi.fn(async (name: string) => ({ name, path: `/skills/${name}/SKILL.md`, content: name === 'alpha' ? '---\nrelated_skills: [beta]\n---\nAlpha body' : 'Beta body' }))
}))
afterEach(() => { cleanup(); queryClient.clear() })

it.each([true, false])('follows a relation through the installed catalog without enabling it (cards=%s)', async cards => {
  $catalogCardView.set(cards)
  queryClient.setQueryData(['public-catalog', 'skills'], [])
  const skills = ['alpha', 'beta'].map(name => ({ name, category: 'test', description: '', enabled: false, provenance: 'agent' as const }))
  render(<QueryClientProvider client={queryClient}>
    <SkillCatalog skills={skills} profile="research" renderInstalledDetail={(skill, onSelectSkill) =>
      <SkillDetail skill={skill} skills={skills} profile="research" onEdit={vi.fn()} onArchive={vi.fn()} onSelectSkill={onSelectSkill} />
    } />
  </QueryClientProvider>)
  if (cards) fireEvent.click(await screen.findByRole('button', { name: 'alpha' }))
  await screen.findByText('Alpha body')
  const heading = screen.getByRole('heading', { name: 'Related skills' })
  fireEvent.click(within(heading.parentElement!).getByRole('button', { name: 'beta' }))
  await screen.findByText('Beta body')
  expect(screen.queryByText('Alpha body')).toBeNull()
  expect(skills.every(skill => !skill.enabled)).toBe(true)
})
