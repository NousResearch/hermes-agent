/**
 * system_admin 对外可见性：列表/查询类接口默认隐藏，仅 bootstrap 密钥或本人可见。
 */
import type { UserEntity, UserPublic } from "../types.js";
import type { V3AuthContext } from "../router/auth.js";

export interface UserVisibilityOptions {
  /** 团队级列表已限定成员集合时，团队 admin 可见其中 normal 用户。 */
  allowTeamPeers?: boolean;
}

export function isSystemAdminUser(user: Pick<UserEntity, "user_type">): boolean {
  return user.user_type === "system_admin";
}

export function canManageUsers(ctx: V3AuthContext): boolean {
  return ctx.isSystemAdmin;
}

/** admin/system_admin 可读全部；普通用户仅可读自己；system_admin 账号对他人默认不可见。 */
export function canViewUser(
  user: UserEntity,
  ctx: V3AuthContext,
  options?: UserVisibilityOptions,
): boolean {
  if (isSystemAdminUser(user)) {
    // 单实例仅一个 system_admin（设计不变量）；ctx.isSystemAdmin 分支在现网不可达，保留作防御。
    return ctx.isAdmin || ctx.isSystemAdmin || ctx.userId === user.user_id;
  }
  if (ctx.isSystemAdmin) return true;
  if (options?.allowTeamPeers) return true;
  if (ctx.isAdmin) return true;
  return ctx.userId === user.user_id;
}

/** v3.1 公开响应：user_id / user_type / username / created_at。 */
export function toPublicUser(user: UserEntity, ctx: V3AuthContext): UserPublic {
  const pub: UserPublic = {
    user_id: user.user_id,
    user_type: user.user_type,
    username: user.username,
    created_at: user.created_at,
  };
  if (isSystemAdminUser(user) && !ctx.isAdmin && ctx.userId !== user.user_id) {
    const { user_type: _ut, ...safe } = pub;
    return safe as UserPublic;
  }
  return pub;
}

export function filterVisibleUsers(
  users: UserEntity[],
  ctx: V3AuthContext,
  options?: UserVisibilityOptions,
): UserPublic[] {
  return users.filter((u) => canViewUser(u, ctx, options)).map((u) => toPublicUser(u, ctx));
}
