import { supabase } from './supabaseClient';

export interface Task {
  id: string;
  title: string;
  status: 'todo' | 'done';
}

export async function fetchTasks(): Promise<Task[]> {
  const { data, error } = await supabase
    .from('tasks')
    .select('*')
    .eq('status', 'todo');
  if (error) throw error;
  return data || [];
}

// Intentional alias import simulation for resolver tests
export { default as ManagerNotice } from '@components/ManagerDesktopNotice';
