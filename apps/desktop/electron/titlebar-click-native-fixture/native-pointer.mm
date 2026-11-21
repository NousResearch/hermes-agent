#import <Cocoa/Cocoa.h>
#include <node_api.h>
#include <string.h>

// In-process NSEvents traverse Cocoa's NSWindow hit testing and Electron's
// native draggable regions. CDP, DOM clicks and sendInputEvent bypass that seam.
// No CGEvent injection, Accessibility permission, activation or cursor movement.
static napi_value Post(napi_env env, napi_callback_info callback) {
  size_t argc = 3;
  napi_value args[3];
  napi_get_cb_info(env, callback, &argc, args, NULL, NULL);
  void *buffer = NULL;
  size_t length = 0;
  double x = 0, y = 0;
  if (argc != 3 || napi_get_buffer_info(env, args[0], &buffer, &length) != napi_ok ||
      length != sizeof(NSView *) || napi_get_value_double(env, args[1], &x) != napi_ok ||
      napi_get_value_double(env, args[2], &y) != napi_ok) {
    napi_throw_type_error(env, NULL, "Expected an Electron native view handle and x/y");
    return NULL;
  }

  void *pointer = NULL;
  memcpy(&pointer, buffer, sizeof(pointer));
  NSView *view = (__bridge NSView *)pointer;
  NSWindow *window = view.window;
  NSPoint local = NSMakePoint(x, view.isFlipped ? y : view.bounds.size.height - y);
  NSPoint point = [view convertPoint:local toView:nil];
  for (NSNumber *type in @[@(NSEventTypeLeftMouseDown), @(NSEventTypeLeftMouseUp)]) {
    NSEvent *event = [NSEvent mouseEventWithType:(NSEventType)type.integerValue
                                      location:point
                                 modifierFlags:0
                                     timestamp:NSProcessInfo.processInfo.systemUptime
                                  windowNumber:window.windowNumber
                                       context:nil
                                   eventNumber:1
                                    clickCount:1
                                      pressure:type.integerValue == NSEventTypeLeftMouseDown ? 1 : 0];
    [NSApp postEvent:event atStart:NO];
  }
  napi_value result;
  napi_get_undefined(env, &result);
  return result;
}

static napi_value DesktopState(napi_env env, napi_callback_info callback) {
  (void)callback;
  NSPoint pointer = NSEvent.mouseLocation;
  NSDictionary *state = @{
    @"frontmostPid": @(NSWorkspace.sharedWorkspace.frontmostApplication.processIdentifier),
    @"pointer": @[@(pointer.x), @(pointer.y)]
  };
  NSData *data = [NSJSONSerialization dataWithJSONObject:state options:0 error:nil];
  NSString *json = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
  napi_value result;
  napi_create_string_utf8(env, json.UTF8String, NAPI_AUTO_LENGTH, &result);
  return result;
}

static napi_value Init(napi_env env, napi_value exports) {
  napi_value fn;
  napi_create_function(env, NULL, 0, Post, NULL, &fn);
  napi_set_named_property(env, exports, "post", fn);
  napi_create_function(env, NULL, 0, DesktopState, NULL, &fn);
  napi_set_named_property(env, exports, "desktopState", fn);
  return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
