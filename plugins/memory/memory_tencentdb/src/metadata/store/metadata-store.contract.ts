/**
 * IMetadataStore 契约测试套件 —— 与后端无关。
 *
 * 对应实施计划「存储切换测试」：同一套用例分别跑在 SQLite / MongoDB 上，
 * 保证不同后端行为一致。SQLite/MongoDB 各自的 *.test.ts 调用 runMetadataStoreContract。
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import type { IMetadataStore } from "./interface.js";
import type { CreateUserInput, CreateTeamInput } from "../types.js";
import { DEFAULT_PAGINATION } from "../pagination.js";
import { newExternalAssetId } from "../utils/external-asset-id.js";
import { buildChatMemoryAssetId } from "../utils/chat-memory-asset.js";

const P = DEFAULT_PAGINATION;

let userSeq = 0;
function uniqueUserInput(over: Partial<CreateUserInput> = {}): CreateUserInput {
  userSeq += 1;
  return {
    auth_provider: "local",
    external_id: `ext-${userSeq}`,
    username: `user${userSeq}`,
    ...over,
  };
}

function teamInput(ownerId: string, over: Partial<CreateTeamInput> = {}): CreateTeamInput {
  return { name: "Team", owner_user_id: ownerId, ...over };
}

function newAssetId(assetType: "skill" | "llm_wiki" | "code_graph" | "chat_memory" = "skill"): string {
  return newExternalAssetId(assetType);
}

/**
 * @param name 后端名称（用于 describe 标题）
 * @param makeStore 每个用例前构造一个干净 store
 * @param teardown 用例后清理
 */
export function runMetadataStoreContract(
  name: string,
  makeStore: () => Promise<IMetadataStore>,
  teardown: (store: IMetadataStore) => Promise<void>,
): void {
  describe(`IMetadataStore contract: ${name}`, () => {
    let store: IMetadataStore;

    beforeEach(async () => {
      store = await makeStore();
      await store.init();
    });

    afterEach(async () => {
      await teardown(store);
    });

    // ── User ──
    describe("User", () => {
      it("createUser 自动生成 user_id / 默认 key 并可按 id/key 查回", async () => {
        const u = await store.createUser(uniqueUserInput());
        expect(u.user_id).toMatch(/^usr-/);
        const defaultKey = await store.getDefaultUserKey(u.user_id);
        expect(defaultKey?.key_value).toBeTruthy();
        expect(await store.getUserById(u.user_id)).toMatchObject({ user_id: u.user_id });
        expect(await store.getUserByKey(defaultKey!.key_value)).toMatchObject({ user_id: u.user_id });
      });

      it("createUser 写入 meta_user_keys 默认行", async () => {
        const u = await store.createUser(uniqueUserInput());
        const keys = (await store.listUserKeys(u.user_id, P)).items;
        const defaultKey = await store.getDefaultUserKey(u.user_id);
        expect(keys.some((k) => k.is_default && k.key_value === defaultKey?.key_value)).toBe(true);
      });

      it("getUserByUsername 命中", async () => {
        const u = await store.createUser(uniqueUserInput({ username: "alice", auth_provider: "local" }));
        const found = await store.getUserByUsername("local", "alice");
        expect(found?.user_id).toBe(u.user_id);
      });

      it("updateUser 局部更新", async () => {
        const u = await store.createUser(uniqueUserInput());
        const updated = await store.updateUser(u.user_id, { display_name: "New Name" });
        expect(updated?.display_name).toBe("New Name");
      });

      it("getUserById 不存在返回 null", async () => {
        expect(await store.getUserById("usr-nope0000")).toBeNull();
      });

      it("deleteUsers 批量删除", async () => {
        const u = await store.createUser(uniqueUserInput());
        const res = await store.deleteUsers([u.user_id, "usr-missing0"]);
        expect(res.deleted_ids).toContain(u.user_id);
        expect(await store.getUserById(u.user_id)).toBeNull();
      });
    });

    // ── Team（自动 admin 成员）──
    describe("Team", () => {
      it("createTeam 自动把 owner 加为 admin 成员（原子）", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        expect(team.team_id).toMatch(/^team-/);

        const members = (await store.listTeamMembers(team.team_id, P)).items;
        expect(members).toHaveLength(1);
        expect(members[0]).toMatchObject({
          user_id: owner.user_id,
          role: "admin",
          status: "active",
        });
      });

      it("listTeamsByUser 返回用户所属 team", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const t1 = await store.createTeam(teamInput(owner.user_id, { name: "T1" }));
        const t2 = await store.createTeam(teamInput(owner.user_id, { name: "T2" }));
        const teams = (await store.listTeamsByUser(owner.user_id, P)).items;
        expect(teams.map((t) => t.team_id).sort()).toEqual([t1.team_id, t2.team_id].sort());
      });

      it("updateTeam / deleteTeams", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const updated = await store.updateTeam(team.team_id, { name: "Renamed" });
        expect(updated?.name).toBe("Renamed");
        const res = await store.deleteTeams([team.team_id]);
        expect(res.deleted_ids).toContain(team.team_id);
        expect(await store.getTeamById(team.team_id)).toBeNull();
      });
    });

    // ── TeamMember ──
    describe("TeamMember", () => {
      it("addTeamMember / getTeamMember / listTeamMembers / removeTeamMember", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const member = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));

        await store.addTeamMember({ team_id: team.team_id, user_id: member.user_id, role: "member" });
        expect(await store.getTeamMember(team.team_id, member.user_id)).toMatchObject({ role: "member" });
        expect((await store.listTeamMembers(team.team_id, P)).items).toHaveLength(2);

        await store.removeTeamMember(team.team_id, member.user_id);
        const remaining = (await store.listTeamMembers(team.team_id, P)).items;
        expect(remaining.map((m) => m.user_id)).not.toContain(member.user_id);
      });

      it("listTeamMembersWithProfile / getTeamMemberWithProfile JOIN username", async () => {
        const owner = await store.createUser(uniqueUserInput({ username: "owner_user" }));
        const member = await store.createUser(uniqueUserInput({ username: "member_user" }));
        const team = await store.createTeam(teamInput(owner.user_id));
        await store.addTeamMember({ team_id: team.team_id, user_id: member.user_id, role: "member" });

        const listed = (await store.listTeamMembersWithProfile(team.team_id, P)).items;
        const byUser = Object.fromEntries(listed.map((m) => [m.user_id, m.username]));
        expect(byUser[owner.user_id]).toBe("owner_user");
        expect(byUser[member.user_id]).toBe("member_user");

        const got = await store.getTeamMemberWithProfile(team.team_id, member.user_id);
        expect(got?.username).toBe("member_user");
      });
    });

    // ── Agent ──
    describe("Agent", () => {
      it("createAgent / get / update / list by team & owner", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agent = await store.createAgent({
          team_id: team.team_id,
          owner_user_id: owner.user_id,
          name: "Coder",
          prompt: "you are coder",
        });
        expect(agent.agent_id).toMatch(/^agt-/);
        expect(agent.visibility).toBe("team");
        expect(agent.status).toBe("active");

        const updated = await store.updateAgent(agent.agent_id, { prompt: "updated" });
        expect(updated?.prompt).toBe("updated");

        expect((await store.listAgentsByTeam(team.team_id, P)).items.map((a) => a.agent_id)).toContain(agent.agent_id);
        expect((await store.listAgentsByOwner(owner.user_id, P)).items.map((a) => a.agent_id)).toContain(agent.agent_id);
      });

      it("listAgentsByTeam 支持 status 过滤", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const a1 = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A1" });
        await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A2", status: "inactive" });
        const active = (await store.listAgentsByTeam(team.team_id, P, { status: "active" })).items;
        expect(active.map((a) => a.agent_id)).toEqual([a1.agent_id]);
      });

      it("listAgentsByTeam 支持 owner_user_id 过滤（组合 team_id）", async () => {
        const u1 = await store.createUser(uniqueUserInput());
        const u2 = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(u1.user_id));
        const a1 = await store.createAgent({ team_id: team.team_id, owner_user_id: u1.user_id, name: "A1" });
        const a2 = await store.createAgent({ team_id: team.team_id, owner_user_id: u2.user_id, name: "A2" });

        // team 全量
        const all = (await store.listAgentsByTeam(team.team_id, P)).items;
        expect(all.map((a) => a.agent_id).sort()).toEqual([a1.agent_id, a2.agent_id].sort());

        // team + owner=u1 → 只返 u1 的
        const mine = (await store.listAgentsByTeam(team.team_id, P, { owner_user_id: u1.user_id })).items;
        expect(mine.map((a) => a.agent_id)).toEqual([a1.agent_id]);

        // team + owner=u2 → 只返 u2 的
        const theirs = (await store.listAgentsByTeam(team.team_id, P, { owner_user_id: u2.user_id })).items;
        expect(theirs.map((a) => a.agent_id)).toEqual([a2.agent_id]);

        // team + owner + status 组合
        await store.createAgent({ team_id: team.team_id, owner_user_id: u1.user_id, name: "A1-inactive", status: "inactive" });
        const activeMine = (
          await store.listAgentsByTeam(team.team_id, P, { owner_user_id: u1.user_id, status: "active" })
        ).items;
        expect(activeMine.map((a) => a.agent_id)).toEqual([a1.agent_id]);
      });
    });

    // ── Task（含 linkAgents 原子）──
    describe("Task", () => {
      it("createTask 含 linked_agents 原子写入 task + task_agents", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agent = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A" });

        const task = await store.createTask({
          team_id: team.team_id,
          creator_user_id: owner.user_id,
          title: "Build feature",
          linked_agents: [{ agent_id: agent.agent_id, role_in_task: "primary" }],
        });
        expect(task.task_id).toMatch(/^task-/);
        const links = (await store.listTaskAgents(task.task_id, P)).items;
        expect(links).toHaveLength(1);
        expect(links[0]).toMatchObject({ agent_id: agent.agent_id, role_in_task: "primary" });
      });

      it("linkTaskAgent / unlinkTaskAgent", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agent = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A" });
        const task = await store.createTask({ team_id: team.team_id, creator_user_id: owner.user_id, title: "T" });

        await store.linkTaskAgent(task.task_id, agent.agent_id, "helper");
        expect((await store.listTaskAgents(task.task_id, P)).items).toHaveLength(1);
        await store.unlinkTaskAgent(task.task_id, agent.agent_id);
        expect((await store.listTaskAgents(task.task_id, P)).items).toHaveLength(0);
      });

      it("listTasksByTeam 支持 status 过滤", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const t1 = await store.createTask({ team_id: team.team_id, creator_user_id: owner.user_id, title: "Run" });
        await store.createTask({ team_id: team.team_id, creator_user_id: owner.user_id, title: "Done", status: "completed" });
        const running = (await store.listTasksByTeam(team.team_id, P, { status: "running" })).items;
        expect(running.map((t) => t.task_id)).toEqual([t1.task_id]);
      });

      it("listTasks 按 creator_user_id 跨 team 查询", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const other = await store.createUser(uniqueUserInput());
        const team1 = await store.createTeam(teamInput(owner.user_id));
        const team2 = await store.createTeam({ name: "T2", owner_user_id: owner.user_id });
        const t1 = await store.createTask({ team_id: team1.team_id, creator_user_id: owner.user_id, title: "A" });
        const t2 = await store.createTask({ team_id: team2.team_id, creator_user_id: owner.user_id, title: "B" });
        await store.createTask({ team_id: team1.team_id, creator_user_id: other.user_id, title: "C" });
        const result = (await store.listTasks({ creator_user_id: owner.user_id }, P)).items;
        expect(result.map((t) => t.task_id).sort()).toEqual([t1.task_id, t2.task_id].sort());
      });
    });

    // ── ParticipationLog ──
    describe("ParticipationLog", () => {
      async function seedParticipationFixture() {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agent = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A" });
        const task = await store.createTask({
          team_id: team.team_id,
          creator_user_id: owner.user_id,
          title: "T",
          linked_agents: [{ agent_id: agent.agent_id }],
        });
        return { owner, team, agent, task };
      }

      it("PL-C1 append 写入；字段完整", async () => {
        const { owner, team, agent, task } = await seedParticipationFixture();
        const log = await store.appendParticipationLog({
          team_id: team.team_id,
          task_id: task.task_id,
          agent_id: agent.agent_id,
          user_id: owner.user_id,
          source: "test",
          metadata_json: '{"session_id":"s1"}',
          created_at: "2026-07-09T10:00:00.000Z",
        });
        expect(log).toMatchObject({
          team_id: team.team_id,
          task_id: task.task_id,
          agent_id: agent.agent_id,
          user_id: owner.user_id,
          source: "test",
          metadata_json: '{"session_id":"s1"}',
          created_at: "2026-07-09T10:00:00.000Z",
          updated_at: "2026-07-09T10:00:00.000Z",
        });
        expect(log.id).toBeTruthy();
      });

      it("PL-C2 list 按 team_id；created_at DESC", async () => {
        const { owner, team, agent, task } = await seedParticipationFixture();
        await store.appendParticipationLog({
          team_id: team.team_id,
          task_id: task.task_id,
          agent_id: agent.agent_id,
          user_id: owner.user_id,
          created_at: "2026-07-09T10:00:00.000Z",
        });
        await store.appendParticipationLog({
          team_id: team.team_id,
          task_id: task.task_id,
          agent_id: agent.agent_id,
          user_id: owner.user_id,
          created_at: "2026-07-09T11:00:00.000Z",
        });
        const page = await store.listParticipationLogs({ team_id: team.team_id }, P);
        expect(page.total).toBe(2);
        expect(page.items.map((l) => l.created_at)).toEqual([
          "2026-07-09T11:00:00.000Z",
          "2026-07-09T10:00:00.000Z",
        ]);
      });

      it("PL-C3 list 叠加 task_id / agent_id / user_id 过滤", async () => {
        const { owner, team, agent, task } = await seedParticipationFixture();
        const other = await store.createUser(uniqueUserInput());
        await store.addTeamMember({ team_id: team.team_id, user_id: other.user_id });
        const otherAgent = await store.createAgent({ team_id: team.team_id, owner_user_id: other.user_id, name: "B" });
        const otherTask = await store.createTask({ team_id: team.team_id, creator_user_id: other.user_id, title: "T2" });
        await store.linkTaskAgent(otherTask.task_id, otherAgent.agent_id);

        await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: owner.user_id,
        });
        await store.appendParticipationLog({
          team_id: team.team_id, task_id: otherTask.task_id, agent_id: otherAgent.agent_id, user_id: other.user_id,
        });

        const filtered = await store.listParticipationLogs({
          team_id: team.team_id,
          task_id: task.task_id,
          agent_id: agent.agent_id,
          user_id: owner.user_id,
        }, P);
        expect(filtered.total).toBe(1);
        expect(filtered.items[0].user_id).toBe(owner.user_id);
      });

      it("PL-C4 list created_after / created_before", async () => {
        const { owner, team, agent, task } = await seedParticipationFixture();
        await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: owner.user_id,
          created_at: "2026-07-01T00:00:00.000Z",
        });
        await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: owner.user_id,
          created_at: "2026-07-15T00:00:00.000Z",
        });
        await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: owner.user_id,
          created_at: "2026-08-01T00:00:00.000Z",
        });
        const page = await store.listParticipationLogs({
          team_id: team.team_id,
          created_after: "2026-07-10T00:00:00.000Z",
          created_before: "2026-07-31T23:59:59.999Z",
        }, P);
        expect(page.total).toBe(1);
        expect(page.items[0].created_at).toBe("2026-07-15T00:00:00.000Z");
      });

      it("PL-C5 list dedupe=true：同 user 多条 → 仅返回最新一条", async () => {
        const { owner, team, agent, task } = await seedParticipationFixture();
        const older = await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: owner.user_id,
          created_at: "2026-07-09T10:00:00.000Z",
        });
        const newer = await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: owner.user_id,
          created_at: "2026-07-09T11:00:00.000Z",
        });
        const page = await store.listParticipationLogs({ team_id: team.team_id, dedupe: true }, P);
        expect(page.items).toHaveLength(1);
        expect(page.items[0].id).toBe(newer.id);
        expect(page.items[0].id).not.toBe(older.id);
      });

      it("PL-C6 list dedupe=true 的 total = 去重用户数", async () => {
        const { owner, team, agent, task } = await seedParticipationFixture();
        const other = await store.createUser(uniqueUserInput());
        await store.addTeamMember({ team_id: team.team_id, user_id: other.user_id });
        await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: owner.user_id,
          created_at: "2026-07-09T10:00:00.000Z",
        });
        await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: owner.user_id,
          created_at: "2026-07-09T11:00:00.000Z",
        });
        await store.appendParticipationLog({
          team_id: team.team_id, task_id: task.task_id, agent_id: agent.agent_id, user_id: other.user_id,
          created_at: "2026-07-09T12:00:00.000Z",
        });
        const page = await store.listParticipationLogs({ team_id: team.team_id, dedupe: true }, P);
        expect(page.total).toBe(2);
        expect(page.items).toHaveLength(2);
      });
    });

    // ── Asset ──
    describe("Asset", () => {
      async function seedAsset() {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const asset = await store.createAsset({
          asset_id: newAssetId(),
          team_id: team.team_id,
          asset_type: "skill",
          name: "Code Review",
          owner_user_id: owner.user_id,
          source_type: "manual",
        });
        return { owner, team, asset };
      }

      it("createAsset 默认值 + get", async () => {
        const { asset } = await seedAsset();
        expect(asset.asset_id).toMatch(/^skl-/);
        expect(asset.version).toBe(1);
        expect(asset.usage_count).toBe(0);
        expect(asset.visibility).toBe("team");
        expect(await store.getAssetById(asset.asset_id)).toMatchObject({ asset_id: asset.asset_id });
      });

      it("touchAssetUsage 累加 usage_count 并更新 last_used_at", async () => {
        const { asset } = await seedAsset();
        await store.touchAssetUsage(asset.asset_id);
        await store.touchAssetUsage(asset.asset_id);
        const got = await store.getAssetById(asset.asset_id);
        expect(got?.usage_count).toBe(2);
        expect(got?.last_used_at).toBeTruthy();
      });

      it("listAssetsByTeam 支持 asset_type / status 过滤", async () => {
        const { team, owner } = await seedAsset();
        await store.createAsset({ asset_id: newAssetId("llm_wiki"), team_id: team.team_id, asset_type: "llm_wiki", name: "Wiki", owner_user_id: owner.user_id, source_type: "manual" });
        const skills = (await store.listAssetsByTeam(team.team_id, P, { asset_type: "skill" })).items;
        expect(skills.every((a) => a.asset_type === "skill")).toBe(true);
        expect(skills).toHaveLength(1);
      });

      it("deleteAssets 物理删除（行消失）+ 幂等", async () => {
        const { asset } = await seedAsset();
        const res = await store.deleteAssets([asset.asset_id]);
        expect(res.deleted_ids).toContain(asset.asset_id);
        expect(await store.getAssetById(asset.asset_id)).toBeNull();
        // 二次删除：已不存在视为成功
        const again = await store.deleteAssets([asset.asset_id]);
        expect(again.deleted_ids).toContain(asset.asset_id);
        expect(again.failed).toEqual([]);
      });
    });

    // ── AgentFixedAsset（全量替换）──
    describe("AgentFixedAsset", () => {
      it("setAgentFixedAssets 全量替换 + list", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agent = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A" });
        const a1 = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "skill", name: "S1", owner_user_id: owner.user_id, source_type: "manual" });
        const a2 = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "skill", name: "S2", owner_user_id: owner.user_id, source_type: "manual" });

        await store.setAgentFixedAssets(agent.agent_id, [
          { asset_id: a1.asset_id, asset_type: "skill", priority: 50, created_by: owner.user_id },
          { asset_id: a2.asset_id, asset_type: "skill", priority: 80, created_by: owner.user_id },
        ]);
        expect((await store.listAgentFixedAssets(agent.agent_id, P)).items).toHaveLength(2);

        // 全量替换为只有 a1
        await store.setAgentFixedAssets(agent.agent_id, [
          { asset_id: a1.asset_id, asset_type: "skill", priority: 10, created_by: owner.user_id },
        ]);
        const after = (await store.listAgentFixedAssets(agent.agent_id, P)).items;
        expect(after).toHaveLength(1);
        expect(after[0]).toMatchObject({ asset_id: a1.asset_id, priority: 10 });
      });

      it("setAgentFixedAssets 传空数组清空绑定", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agent = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A" });
        const a1 = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "skill", name: "S1", owner_user_id: owner.user_id, source_type: "manual" });
        await store.setAgentFixedAssets(agent.agent_id, [{ asset_id: a1.asset_id, asset_type: "skill", created_by: owner.user_id }]);
        await store.setAgentFixedAssets(agent.agent_id, []);
        expect((await store.listAgentFixedAssets(agent.agent_id, P)).items).toHaveLength(0);
      });

      it("summarizeAgentFixedAssetsByAgents 按 type 聚合 + asset_id 过滤", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const a1 = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A1" });
        const a2 = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A2" });
        const skill = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "skill", name: "S", owner_user_id: owner.user_id, source_type: "manual" });
        const wiki = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "llm_wiki", name: "W", owner_user_id: owner.user_id, source_type: "manual" });
        const cg = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "code_graph", name: "C", owner_user_id: owner.user_id, source_type: "manual" });

        await store.setAgentFixedAssets(a1.agent_id, [
          { asset_id: skill.asset_id, asset_type: "skill", created_by: owner.user_id },
          { asset_id: wiki.asset_id, asset_type: "llm_wiki", created_by: owner.user_id },
          { asset_id: cg.asset_id, asset_type: "code_graph", created_by: owner.user_id },
        ]);
        await store.setAgentFixedAssets(a2.agent_id, [
          { asset_id: wiki.asset_id, asset_type: "llm_wiki", created_by: owner.user_id },
        ]);

        const rows = await store.summarizeAgentFixedAssetsByAgents([a1.agent_id, a2.agent_id, "agt-missing"]);
        expect(rows).toEqual(
          expect.arrayContaining([
            { agent_id: a1.agent_id, asset_type: "skill", cnt: 1 },
            { agent_id: a1.agent_id, asset_type: "llm_wiki", cnt: 1 },
            { agent_id: a1.agent_id, asset_type: "code_graph", cnt: 1 },
            { agent_id: a2.agent_id, asset_type: "llm_wiki", cnt: 1 },
          ]),
        );
        expect(rows.some((r) => r.agent_id === "agt-missing")).toBe(false);

        const filtered = await store.summarizeAgentFixedAssetsByAgents(
          [a1.agent_id, a2.agent_id],
          { assetId: wiki.asset_id },
        );
        expect(filtered).toHaveLength(2);
        expect(filtered.every((r) => r.asset_type === "llm_wiki" && r.cnt === 1)).toBe(true);
      });
    });

    // ── ACL ──
    describe("ACL", () => {
      it("grantAcl / listAclByAsset / listAclBySubject / revokeAcl", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const asset = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "skill", name: "S", owner_user_id: owner.user_id, source_type: "manual" });

        const acl = await store.grantAcl({
          asset_id: asset.asset_id,
          subject_type: "user",
          subject_id: owner.user_id,
          permission: "write",
          granted_by: owner.user_id,
        });
        expect(acl.effect).toBe("allow");
        expect((await store.listAclByAsset(asset.asset_id, P)).items).toHaveLength(1);
        expect((await store.listAclBySubject("user", owner.user_id, P)).items).toHaveLength(1);

        await store.revokeAcl(acl.id);
        expect((await store.listAclByAsset(asset.asset_id, P)).items).toHaveLength(0);
      });
    });

    // ── Delete Cascade (N1) ──
    describe("Delete Cascade", () => {
      it("deleteUsers 级联清理 team_members + ACL", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const asset = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "skill", name: "S", owner_user_id: owner.user_id, source_type: "manual" });
        await store.grantAcl({ asset_id: asset.asset_id, subject_type: "user", subject_id: owner.user_id, permission: "write", granted_by: owner.user_id });

        expect((await store.listTeamMembers(team.team_id, P)).items).toHaveLength(1);
        expect((await store.listAclBySubject("user", owner.user_id, P)).items).toHaveLength(1);

        await store.deleteUsers([owner.user_id]);
        expect((await store.listTeamMembers(team.team_id, P)).items).toHaveLength(0);
        expect((await store.listAclBySubject("user", owner.user_id, P)).items).toHaveLength(0);
      });

      it("deleteAgents 级联清理 task_agents + fixed_assets", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agent = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A" });
        const task = await store.createTask({ team_id: team.team_id, creator_user_id: owner.user_id, title: "T" });
        const asset = await store.createAsset({ asset_id: newAssetId(), team_id: team.team_id, asset_type: "skill", name: "S", owner_user_id: owner.user_id, source_type: "manual" });
        await store.linkTaskAgent(task.task_id, agent.agent_id);
        await store.setAgentFixedAssets(agent.agent_id, [{ asset_id: asset.asset_id, asset_type: "skill", created_by: owner.user_id }]);

        await store.deleteAgents([agent.agent_id]);
        expect((await store.listTaskAgents(task.task_id, P)).items).toHaveLength(0);
        expect((await store.listAgentFixedAssets(agent.agent_id, P)).items).toHaveLength(0);
      });

      it("deleteAgents 归档自身 chat_memory 并清理其它 agent 的借入绑定", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agentA = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A" });
        const agentB = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "B" });
        const selfMemoryA = buildChatMemoryAssetId(team.team_id, agentA.agent_id);
        const selfMemoryB = buildChatMemoryAssetId(team.team_id, agentB.agent_id);
        await store.createAsset({
          asset_id: selfMemoryA,
          team_id: team.team_id,
          asset_type: "chat_memory",
          name: "Memory of A",
          owner_user_id: owner.user_id,
          source_type: "auto",
          visibility: "team",
          status: "active",
        });
        await store.createAsset({
          asset_id: selfMemoryB,
          team_id: team.team_id,
          asset_type: "chat_memory",
          name: "Memory of B",
          owner_user_id: owner.user_id,
          source_type: "auto",
          visibility: "private",
          status: "active",
        });
        await store.setAgentFixedAssets(agentB.agent_id, [
          { asset_id: selfMemoryB, asset_type: "chat_memory", created_by: owner.user_id },
          { asset_id: selfMemoryA, asset_type: "chat_memory", created_by: owner.user_id },
        ]);

        await store.deleteAgents([agentA.agent_id]);

        expect(await store.getAssetById(selfMemoryA)).toBeNull();
        const remaining = (await store.listAgentFixedAssets(agentB.agent_id, P)).items.map((b) => b.asset_id);
        expect(remaining).toHaveLength(1);
        expect(remaining).toContain(selfMemoryB);
        const rows = await store.summarizeAgentFixedAssetsByAgents([agentB.agent_id]);
        expect(rows).toEqual([{ agent_id: agentB.agent_id, asset_type: "chat_memory", cnt: 1 }]);
      });

      it("deleteTasks 级联清理 task_agents", async () => {
        const owner = await store.createUser(uniqueUserInput());
        const team = await store.createTeam(teamInput(owner.user_id));
        const agent = await store.createAgent({ team_id: team.team_id, owner_user_id: owner.user_id, name: "A" });
        const task = await store.createTask({ team_id: team.team_id, creator_user_id: owner.user_id, title: "T" });
        await store.linkTaskAgent(task.task_id, agent.agent_id);

        await store.deleteTasks([task.task_id]);
        expect((await store.listTaskAgents(task.task_id, P)).items).toHaveLength(0);
      });
    });

    // ── v3.1：username / external_id 无唯一约束 ──
    describe("User uniqueness (v3.1)", () => {
      it("相同 username 可重复创建", async () => {
        const username = `dup_${Math.random()}`;
        const a = await Promise.resolve(store.createUser(uniqueUserInput({ username })));
        const b = await Promise.resolve(
          store.createUser(uniqueUserInput({ username, external_id: `other_${Math.random()}` })),
        );
        expect(a.user_id).not.toBe(b.user_id);
      });
    });

    // ── ConfigParam ──
    describe("ConfigParam", () => {
      it("upsertConfigParam inserts global param and getConfigParam retrieves it", async () => {
        const entity = await store.upsertConfigParam({
          scope: "global",
          user_id: null,
          module: "quota",
          param_name: "max_users_per_instance",
          param_value: "500",
          description: "实例用户上限",
        });
        expect(entity.id).toBeGreaterThan(0);
        expect(entity.scope).toBe("global");
        expect(entity.user_id).toBeNull();
        expect(entity.module).toBe("quota");
        expect(entity.param_name).toBe("max_users_per_instance");
        expect(entity.param_value).toBe("500");

        const found = await store.getConfigParam("global", null, "quota", "max_users_per_instance");
        expect(found).not.toBeNull();
        expect(found!.param_value).toBe("500");
      });

      it("upsertConfigParam updates existing global param", async () => {
        await store.upsertConfigParam({
          scope: "global",
          user_id: null,
          module: "quota",
          param_name: "max_teams_per_instance",
          param_value: "100",
          description: "实例团队上限",
        });
        const updated = await store.upsertConfigParam({
          scope: "global",
          user_id: null,
          module: "quota",
          param_name: "max_teams_per_instance",
          param_value: "200",
          description: "实例团队上限（已调整）",
        });
        expect(updated.param_value).toBe("200");
        expect(updated.description).toBe("实例团队上限（已调整）");
      });

      it("upsertConfigParam inserts user-level param", async () => {
        const user = await store.createUser(uniqueUserInput());
        const entity = await store.upsertConfigParam({
          scope: "user",
          user_id: user.user_id,
          module: "asset_type",
          param_name: "skill.enabled",
          param_value: "0",
          description: "Skill 开关",
        });
        expect(entity.scope).toBe("user");
        expect(entity.user_id).toBe(user.user_id);
        expect(entity.param_value).toBe("0");

        const found = await store.getConfigParam("user", user.user_id, "asset_type", "skill.enabled");
        expect(found).not.toBeNull();
        expect(found!.param_value).toBe("0");
      });

      it("upsertConfigParam updates existing user-level param", async () => {
        const user = await store.createUser(uniqueUserInput());
        await store.upsertConfigParam({
          scope: "user",
          user_id: user.user_id,
          module: "asset_type",
          param_name: "skill.enabled",
          param_value: "0",
          description: "Skill 开关",
        });
        const updated = await store.upsertConfigParam({
          scope: "user",
          user_id: user.user_id,
          module: "asset_type",
          param_name: "skill.enabled",
          param_value: "1",
          description: "Skill 开关",
        });
        expect(updated.param_value).toBe("1");
      });

      it("getConfigParam returns null for non-existent", async () => {
        const found = await store.getConfigParam("global", null, "quota", "nonexistent");
        expect(found).toBeNull();
      });

      it("listConfigParams filters by module", async () => {
        await store.upsertConfigParam({
          scope: "global", user_id: null, module: "quota",
          param_name: "max_users_per_instance", param_value: "500", description: "d1",
        });
        await store.upsertConfigParam({
          scope: "global", user_id: null, module: "quota",
          param_name: "max_teams_per_instance", param_value: "100", description: "d2",
        });
        await store.upsertConfigParam({
          scope: "global", user_id: null, module: "asset_type",
          param_name: "skill.enabled", param_value: "1", description: "d3",
        });

        const quota = await store.listConfigParams({ module: "quota" });
        expect(quota).toHaveLength(2);
        expect(quota.every((r) => r.module === "quota")).toBe(true);
      });

      it("listConfigParams filters by userId returns global + user rows", async () => {
        const user = await store.createUser(uniqueUserInput());
        await store.upsertConfigParam({
          scope: "global", user_id: null, module: "asset_type",
          param_name: "skill.enabled", param_value: "1", description: "d",
        });
        await store.upsertConfigParam({
          scope: "user", user_id: user.user_id, module: "asset_type",
          param_name: "skill.enabled", param_value: "0", description: "d",
        });

        const rows = await store.listConfigParams({ module: "asset_type", userId: user.user_id });
        expect(rows.length).toBeGreaterThanOrEqual(2);
        const scopes = rows.map((r) => r.scope);
        expect(scopes).toContain("global");
        expect(scopes).toContain("user");
      });

      it("listConfigParams filters by paramNames", async () => {
        await store.upsertConfigParam({
          scope: "global", user_id: null, module: "asset_type",
          param_name: "skill.enabled", param_value: "1", description: "d",
        });
        await store.upsertConfigParam({
          scope: "global", user_id: null, module: "asset_type",
          param_name: "llm_wiki.enabled", param_value: "1", description: "d",
        });
        await store.upsertConfigParam({
          scope: "global", user_id: null, module: "asset_type",
          param_name: "code_graph.enabled", param_value: "1", description: "d",
        });

        const rows = await store.listConfigParams({
          module: "asset_type",
          paramNames: ["skill.enabled", "llm_wiki.enabled"],
        });
        expect(rows).toHaveLength(2);
      });

      it("user-level and global params are isolated (different scope)", async () => {
        const user = await store.createUser(uniqueUserInput());
        await store.upsertConfigParam({
          scope: "global", user_id: null, module: "asset_type",
          param_name: "skill.enabled", param_value: "1", description: "global",
        });
        await store.upsertConfigParam({
          scope: "user", user_id: user.user_id, module: "asset_type",
          param_name: "skill.enabled", param_value: "0", description: "user",
        });

        const global = await store.getConfigParam("global", null, "asset_type", "skill.enabled");
        const userLevel = await store.getConfigParam("user", user.user_id, "asset_type", "skill.enabled");
        expect(global!.param_value).toBe("1");
        expect(userLevel!.param_value).toBe("0");
      });
    });
  });
}
