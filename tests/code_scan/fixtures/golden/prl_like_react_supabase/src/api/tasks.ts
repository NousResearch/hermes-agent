import { supabase } from '../supabaseClient';

interface FixtureTask {
  id: number;
  title: string;
  status?: string;
}

export async function loadWorksiteTasks(): Promise<FixtureTask[]> {
  const { data } = await supabase.from('fixture_tasks').select('id,title,status');
  return data ?? [];
}

export function summarizeTaskStatus(tasks: FixtureTask[]): Record<string, number> {
  return tasks.reduce<Record<string, number>>((counts, task) => {
    const status = task.status ?? 'unknown';
    counts[status] = (counts[status] ?? 0) + 1;
    return counts;
  }, {});
}
