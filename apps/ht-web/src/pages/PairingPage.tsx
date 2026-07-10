import { useState } from "react";
import {
  approvePairing,
  clearPendingPairing,
  getPairing,
  revokePairing,
  type PairingUser,
} from "@/api/pairing";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";

export default function PairingPage() {
  const pairing = useResource(getPairing);
  const [busy, setBusy] = useState(false);

  const clearPending = async () => {
    setBusy(true);
    try {
      await clearPendingPairing();
      pairing.reload();
    } finally {
      setBusy(false);
    }
  };

  return (
    <ManagementPage
      title="Pairing"
      actions={
        <button type="button" className="ht-btn ht-btn--sm ht-btn--ghost" onClick={pairing.reload}>
          Refresh
        </button>
      }
    >
      <ResourceView resource={pairing}>
        {(d) => (
          <>
            <div className="ht-card">
              <div className="ht-card__title">Pending requests</div>
              {d.pending.length === 0 ? (
                <p className="ht-muted">No pending requests.</p>
              ) : (
                <>
                  <table className="ht-table">
                    <thead>
                      <tr>
                        <th>Platform</th>
                        <th>User</th>
                        <th>Code</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {d.pending.map((u) => (
                        <PendingRow
                          key={`${u.platform}:${u.user_id}`}
                          user={u}
                          onChanged={pairing.reload}
                        />
                      ))}
                    </tbody>
                  </table>
                  <div className="ht-row-actions">
                    <button
                      type="button"
                      className="ht-btn ht-btn--sm ht-btn--stop"
                      disabled={busy}
                      onClick={clearPending}
                    >
                      Clear all pending
                    </button>
                  </div>
                </>
              )}
            </div>

            <div className="ht-card">
              <div className="ht-card__title">Paired devices</div>
              {d.approved.length === 0 ? (
                <p className="ht-muted">No paired devices.</p>
              ) : (
                <table className="ht-table">
                  <thead>
                    <tr>
                      <th>Platform</th>
                      <th>User</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {d.approved.map((u) => (
                      <ApprovedRow
                        key={`${u.platform}:${u.user_id}`}
                        user={u}
                        onChanged={pairing.reload}
                      />
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

function userLabel(u: PairingUser) {
  return u.user_name || u.user_id;
}

function PendingRow({ user, onChanged }: { user: PairingUser; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);

  const approve = async () => {
    if (!user.code) return;
    setBusy(true);
    try {
      await approvePairing(user.platform, user.code);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const deny = async () => {
    setBusy(true);
    try {
      await revokePairing(user.platform, user.user_id);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <tr>
      <td>
        <span className="ht-chip">{user.platform}</span>
      </td>
      <td>
        {userLabel(user)}
        {typeof user.age_minutes === "number" && (
          <div className="ht-muted ht-sm">{user.age_minutes}m ago</div>
        )}
      </td>
      <td>{user.code ? <code>{user.code}</code> : <span className="ht-dim">—</span>}</td>
      <td className="ht-row-actions">
        <button
          type="button"
          className="ht-btn ht-btn--sm"
          disabled={busy || !user.code}
          onClick={approve}
        >
          Approve
        </button>
        <button
          type="button"
          className="ht-btn ht-btn--sm ht-btn--stop"
          disabled={busy}
          onClick={deny}
        >
          Deny
        </button>
      </td>
    </tr>
  );
}

function ApprovedRow({ user, onChanged }: { user: PairingUser; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);

  const revoke = async () => {
    setBusy(true);
    try {
      await revokePairing(user.platform, user.user_id);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <tr>
      <td>
        <span className="ht-chip">{user.platform}</span>
      </td>
      <td>{userLabel(user)}</td>
      <td className="ht-row-actions">
        <button
          type="button"
          className="ht-btn ht-btn--sm ht-btn--stop"
          disabled={busy}
          onClick={revoke}
        >
          Revoke
        </button>
      </td>
    </tr>
  );
}
