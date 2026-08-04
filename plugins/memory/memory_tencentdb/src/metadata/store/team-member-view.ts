import type { MemberStatus, TeamMemberEntity, TeamMemberView, TeamRole } from "../types.js";

type TeamMemberProfileRow = TeamMemberEntity & { username?: string | null };

export function mapTeamMemberWithProfile(row: TeamMemberProfileRow): TeamMemberView {
  return {
    id: row.id,
    team_id: row.team_id,
    user_id: row.user_id,
    role: row.role as TeamRole,
    joined_at: row.joined_at,
    status: row.status as MemberStatus,
    username: row.username ?? "",
  };
}
