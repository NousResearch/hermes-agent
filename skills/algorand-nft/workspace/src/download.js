export function saveBytes(document,name,bytes,type,urlAPI=URL,defer=fn=>setTimeout(fn,1000)) {
  const url=urlAPI.createObjectURL(new Blob([bytes],{type}));
  const link=document.createElement('a');link.href=url;link.download=name;
  document.body.append(link);
  try {link.click();} finally {link.remove();defer(()=>urlAPI.revokeObjectURL(url));}
}
